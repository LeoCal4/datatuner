import json
import logging
import os
from collections import defaultdict
from pyexpat import model
from typing import List

import torch
from datatuner.lm.converters import converters
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

logger = logging.getLogger(__file__)

PAD_TOKEN = "<pad>"
GPT2_MODEL_INPUTS = ["input_ids", "mc_token_ids", "lm_labels", "mc_labels", "token_type_ids"]
T5_MODEL_INPUTS = ["input_ids", "lm_labels", "attention_mask"]
GPT2_PADDED_INPUTS = ["input_ids", "lm_labels", "token_type_ids"]
T5_PADDED_INPUTS = ["input_ids", "lm_labels"]

MASKED_OUTPUT = -1


def get_model_inputs(model_type: str) -> List[str]:
    model_type = str(model_type)
    if "gpt2" in model_type:
        return GPT2_MODEL_INPUTS
    elif "t5" in model_type:
        return T5_MODEL_INPUTS
    else:
        raise ValueError(f"Model type/checkpoint {model_type} is not valid.")


def get_padded_inputs(model_type: str) -> List[str]:
    model_type = str(model_type)
    if "gpt2" in model_type:
        return GPT2_PADDED_INPUTS
    elif "t5" in model_type:
        return T5_PADDED_INPUTS
    else:
        raise ValueError(f"Model type/checkpoint {model_type} is not valid.")

def build_input_from_segments(
        data_point,
        tokenizer,
        task_config,
        with_eos=True,
        mask_lm_labels=False,
        candidate_val=None, # the actual sentence 
        max_block_size=None,
):
    """Builds the complete input of the model for a given data point, iterating on the task_config fields.
    This expedient is used so that during evaluation the text part can be avoided.

    Args:
        data_point (dict): processed data point, tokenized and "id-ized" 
        tokenizer
        task_config (dict): tokenization configuration data
        with_eos (bool, optional): defaults to True.
        mask_lm_labels (bool, optional): if the labels needs to be totally masked or not (only used for distractors). Defaults to False.
        candidate_val (_type_, optional): the original sentence from which the data was extracted. Defaults to None.

    Returns:
        dict: the final input data point comprising of input_type_ids, token_type_ids and labels
    """
    instance = {}
    sequence, token_types, lm_labels = [], [], []
    curr_span_type = 0

    # TODO: change this to be the max of the current tokenizer by name, not min of all maxes
    max_tokenizer_size = min(tokenizer.max_model_input_sizes.values())
    if max_block_size is not None:
        max_tokenizer_size = min(max_block_size, max_tokenizer_size)

    #* iterate on the entries of the data_shape field in the tokenization config,
    #*    which represents how the input data should be represented for this task 
    for item in task_config["data_shape"]:
        #* add the token for the special tag (i.e. the one for <data>) and set the token type
        if item["type"] == "special":
            x = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(item["id"]))
            # this gets updated only when a special token in encountered, since they indicate
            #     the two different parts of a data entry (data or text)
            curr_span_type = x[0]
            tokens = x
        #* not used
        elif item["type"] == "special_id":
            tokens = item["id"]
        #* add the text contained in the field named as the current iterate (i.e new_mr, ref...)
        elif item["type"] == "text":
            # if we are using the DoubleHeads setting, we might have the input as a list of texts
            # the candidate_val contains the tokens of the item which we consider now
            if data_point[item["id"]] and type(data_point[item["id"]][0]) == list:
                #* if its a list (hence the distractors + the original sentence) take the original sentence
                tokens = candidate_val
            else:
                #* otherwise access the field and take its text
                tokens = data_point[item["id"]]
        else:
            raise Exception("Invalid item type in the data shape")

        #* update the sentence and the token_types
        sequence += tokens
        current_token_types = [curr_span_type] * len(tokens)

        #* non-coarse grained means that the token_type_ids are not just set to distinguish 
        #*     the main two types in the data points (data or text), but also to represent the 
        #*     the "smaller" ones (i.e <available_on_steam>...)
        if "token_typing" not in task_config or task_config["token_typing"] != "coarse_grained":
            # if we have special tokens within the tokens, we adjust the token_type_ids so that anything after
            # a special token has the token_type as the id of that special token.
            for t_i, token in enumerate(tokens):
                if token in tokenizer.added_tokens_decoder:
                    curr_span_type = token
                current_token_types[t_i] = curr_span_type

        token_types += current_token_types
        #* add the tokens as labels only if the current type needs to be learned (for example "ref" in viggo),
        #*    otherwise use mask
        lm_labels += tokens if item["learn"] else ([MASKED_OUTPUT] * len(tokens))

    if with_eos:
        eos = tokenizer.convert_tokens_to_ids(["<eos>"])
        sequence += eos
        token_types += [curr_span_type]
        lm_labels += eos

    sequence = sequence[:max_tokenizer_size]
    token_types = token_types[:max_tokenizer_size]
    lm_labels = lm_labels[:max_tokenizer_size]

    assert len(sequence) == len(token_types)
    assert len(token_types) == len(lm_labels)

    instance["input_ids"] = sequence
    instance["token_type_ids"] = token_types

    if mask_lm_labels: # masked if the current sentence is a distractor
        instance["lm_labels"] = [MASKED_OUTPUT] * len(instance["input_ids"])
    else:
        instance["lm_labels"] = lm_labels

    return instance, sequence


def get_inputs(item, device, tokenizer, task_config):
    """Get the input_ids and the token_type_ids from the item dictionary"""
    instance, _ = build_input_from_segments(item, tokenizer, task_config, with_eos=False)
    input_ids = torch.tensor(instance["input_ids"], device=device).unsqueeze(0)
    token_type_ids = torch.tensor(instance["token_type_ids"], device=device).unsqueeze(0)
    return input_ids, token_type_ids


def pad_dataset(dataset, inputs_to_pad, padding=0, pad_validation_left=False, create_attention_mask=False):
    """Pad the dataset. This could be optimized by defining a
    Dataset class and padd only batches but this is simpler. (<- I hate you)

    Use the create_attention_mask flag to add the attention_mask field too (this is terrible)"""
    max_l = max(len(x) for x in dataset["input_ids"])
    if create_attention_mask:
        dataset["attention_mask"] = []
    for name in inputs_to_pad:
        if name not in dataset:
            continue
        padding_token = [padding if name != "lm_labels" else MASKED_OUTPUT]
        new_dataset_current_list = []
        for entry in dataset[name]:
            if name == "validation" and pad_validation_left:
                padding_first_part = padding_token * (max_l - len(entry))
                padding_second_part = entry
                if create_attention_mask and name == "input_ids":
                    attn_mask_first_part = [0] * (max_l - len(entry))
                    attn_mask_second_part = [1] * len(entry)
            else:
                padding_first_part = entry
                padding_second_part = padding_token * (max_l - len(entry)) 
                if create_attention_mask and name == "input_ids":
                    attn_mask_first_part = [1] * len(entry)
                    attn_mask_second_part = [0] * (max_l - len(entry)) 
            if create_attention_mask and name == "input_ids":
                dataset["attention_mask"].append(attn_mask_first_part + attn_mask_second_part)
            new_dataset_current_list.append(padding_first_part + padding_second_part)
        dataset[name] = new_dataset_current_list
    return dataset


def get_data_loaders(args, task_config, tokenizer):
    """ Prepare the dataset for training and evaluation """
    global MASKED_OUTPUT
    logger.info("Loading training data")

    #* set mask value to -100
    if args.use_custom_t5 or "t5" in args.model_type:
        MASKED_OUTPUT = -100

    #* Make sure only the first process in distributed training will download model & vocab
    if args.local_rank not in [-1, 0]:
        torch.distributed.barrier()
        args.ignore_cache = False

    datasets_raw = {}
    for split in ["validation", "train"]:
        logger.info(f"Loading {split} data")
        datasets_raw[split] = get_dataset(
            tokenizer,
            args.dataset_cache,
            task_config,
            args.dataset_path,
            split,
            args.max_data if split == "train" else args.val_max_data,
            args.ignore_cache,
            args.max_block_size,
        )

    logger.info("Build inputs and labels")
    datasets = {"train": defaultdict(list), "validation": defaultdict(list)}

    for dataset_name, dataset in datasets_raw.items():
        #* get the last learnt field aka the field containing the original sentence and its distractors, i.e "ref" for viggo
        last_learnt_field = [x["id"] for x in task_config["data_shape"][::-1] if x["learn"] and x["type"] == "text"][0]

        #* set the number of candidates
        if args.multitask:
            assert type(dataset[0][last_learnt_field]) == list
            num_candidates = len(dataset[0][last_learnt_field])
        else:
            num_candidates = 1
        if args.num_candidates > 0 and dataset_name in ["train", "validation"]:
            num_candidates = min(args.num_candidates, num_candidates)

        for data_point in dataset:
            # in case there is only one target sentence, turn it into a list
            if type(data_point[last_learnt_field]) == str:
                data_point[last_learnt_field] = [data_point[last_learnt_field]]

            # candidate_val is the original sentence (or the distractor if num_candidates > 1)
            for j, candidate_val in enumerate(data_point[last_learnt_field][-num_candidates:]):
                # the last item in the array is the ground truth. For other distractor items, we mask the LM labels
                mask_lm_labels = bool(j != num_candidates - 1)
                instance, _ = build_input_from_segments(
                    data_point,
                    tokenizer,
                    task_config,
                    mask_lm_labels=mask_lm_labels,
                    candidate_val=candidate_val,
                    max_block_size=args.max_block_size,
                )
                if args.multitask:
                    # this is an indicator for the last input token, used in the Double Head model
                    instance["mc_token_ids"] = len(instance["input_ids"]) - 1

                #* append the new input_ids, token_type_ids and labels
                for input_name, input_array in instance.items():
                    if args.use_custom_t5 or "t5" in args.model_type:  # TODO add a way to make this cleaner
                            if input_name == "token_type_ids":
                                continue
                    datasets[dataset_name][input_name].append(input_array)

            datasets[dataset_name]["n_candidates"] = num_candidates

            # the ground truth is the last item in the array; previous items are distractors
            if args.multitask:
                datasets[dataset_name]["mc_labels"].append(num_candidates - 1)

    logger.info("Pad inputs and convert to Tensor")
    # logger.info(f"{args}")
    tensor_datasets = {"train": [], "validation": []}
    for dataset_name, dataset in datasets.items():
        dataset = pad_dataset(
            dataset, get_padded_inputs(args.model_checkpoint), padding=tokenizer.convert_tokens_to_ids(PAD_TOKEN), 
            pad_validation_left=False, create_attention_mask=args.use_custom_t5) #! revert pad_validation_left value
        model_inputs = get_model_inputs(args.model_checkpoint)
        for input_name in model_inputs:
            if input_name in dataset:
                tensor = torch.tensor(dataset[input_name])
                if input_name != "mc_labels" and not args.use_custom_t5: # TODO fix the shape
                    # adjust the shape as we might have more than one candidate in the case of DoubleHeads
                    # this adds a dimension which is normally just mono-dimensional, hence useless
                    tensor = tensor.view((-1, datasets[dataset_name]["n_candidates"]) + tensor.shape[1:])
                tensor_datasets[dataset_name].append(tensor)

    logger.info("Build train and validation dataloaders")
    train_dataset, valid_dataset = (
        TensorDataset(*tensor_datasets["train"]),
        TensorDataset(*tensor_datasets["validation"]),
    )
    train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset) if args.distributed else None
    valid_sampler = torch.utils.data.distributed.DistributedSampler(valid_dataset) if args.distributed else None
    train_loader = DataLoader(
        train_dataset, sampler=train_sampler, batch_size=args.train_batch_size, shuffle=(not args.distributed)
    )
    valid_loader = DataLoader(valid_dataset, sampler=valid_sampler, batch_size=args.valid_batch_size, shuffle=False)

    logger.info("Train dataset (Batch, Candidates, Seq length): {}".format(train_dataset.tensors[0].shape))
    logger.info("validation dataset (Batch, Candidates, Seq length): {}".format(valid_dataset.tensors[0].shape))

    if args.local_rank == 0:
        # Make sure only the first process in distributed training will download model & vocab
        torch.distributed.barrier()

    return train_loader, valid_loader, train_sampler, valid_sampler


def get_dataset_from_file(tokenizer, filename, task_config, max_data, max_block_size=None):
    """Opens the dataset indicated from the given filename, then tokenizes and "id-izes" (converts to ids)
    all the text fields of each data point, truncating their length if necessary.

    Args:
        tokenizer
        filename (str): file name of the dataset
        task_config (dict): tokenization configuration data
        max_data (int): the amount of data points wanted from the dataset.
        max_block_size (int, optional): defaults to None.
    """

    def tokenize(obj):
        """If obj is a string, tokenize it and convert the tokens to ids.
        If obj is a dict or an iterable, recursively call this method on each string composing them.

        Args:
            obj (str, dict or iterable)

        Returns:
            list: list of token_ids 
        """
        if isinstance(obj, str):
            return tokenizer.convert_tokens_to_ids(tokenizer.tokenize(obj))
        if isinstance(obj, dict):
            return dict((n, tokenize(o)) for n, o in obj.items())
        return list(tokenize(o) for o in obj)

    #* open the dataset file
    with open(filename, "r") as f:
        data = json.load(f)

    #* get the max size supported by the tokenizer model
    # {'gpt2': 1024, 'gpt2-medium': 1024, 'gpt2-large': 1024, 'distilgpt2': 1024}
    max_tokenizer_size = min(tokenizer.max_model_input_sizes.values())
    if max_block_size is not None:
        max_tokenizer_size = min(max_block_size, max_tokenizer_size)

    if max_data > 0:
        data = data[:max_data]

    ignored_sequences = 0

    output_data = []
    logger.info(f"initial data: {len(data)}")

    #* get the name of the text fields 
    text_fields = [x for x in task_config["data_shape"] if x["type"] == "text"]

    #* get the lengths of the special fields' names
    len_special_fields = 0
    for x in task_config["data_shape"]:
        if x["type"] == "special":
            len_special_fields += len(tokenizer.tokenize(x["id"]))
        elif x["type"] == "special_id":
            len_special_fields += len(x["id"])

    failed_conversions = 0
    for inst_i, inst in enumerate(tqdm(data)): # tqdm is just the progress bar iterator
        # check the inclusion criteria 
        #! didn't find this anywhere
        if "include" in task_config:
            include = True
            for field, value in task_config["include"].items():
                if field in inst and inst[field] != value:
                    include = False
                    break
            if not include:
                continue

        item = {}
        total_seq_len = 0
        stop = False
        #* iterate on the text field of the current data point
        for field in text_fields:
            field_v = inst[field["id"]]

            #! didn't find this anywhere too
            if "converter" in field:
                try:
                    func = converters[field["converter"]]
                except:
                    logger.error(f"Unable to get the converter {field['converter']}")
                    raise
                field_v = func(field_v)
                if field_v is None:
                    stop = True
                    break
            
            #* tokenize the text field and update the total sequence length
            item[field["id"]] = tokenize(field_v)
            total_seq_len += len(item[field["id"]])

        if stop:
            failed_conversions += 1
            continue
        
        #* add any needed extra_field
        if "extra_fields" in task_config: # found in webnlg
            for field in task_config["extra_fields"]:
                item[field] = inst[field]

        #* truncate the text fields of the data point if it is greater than the max input size acceptable from the model (?)
        if total_seq_len + len_special_fields + 1 > max_tokenizer_size: # 1 is for eos token
            for field in text_fields:
                item[field["id"]] = item[field["id"]][: max_tokenizer_size - 100] #? why -100?
            print(f"warning: this input is longer than the sequence length so we truncated: {inst_i}")
            ignored_sequences += 1
            # continue
        output_data.append(item)

    logger.info(
        "%d / %d sequences ignored due to positional embedding restriction or max block size restriction"
        % (ignored_sequences, len(data))
    )
    logger.info("%d / %d removed due to failed conversions" % (failed_conversions, len(data)))
    logger.info(f"preprocessed data: {len(output_data)}")
    return output_data


def get_dataset(tokenizer, dataset_cache, task_config, path, split, max_data, ignore_cache, max_block_size):
    """Load and processes the dataset for the given split"""
    dataset_cache = (
        f"{dataset_cache}_{split}_{task_config['name']}_{max_data}_{type(tokenizer).__name__}"
    )  # Do avoid using GPT cache for GPT-2 and vice-versa

    if not ignore_cache and dataset_cache and os.path.isfile(dataset_cache):
        logger.info("Load tokenized dataset from cache at %s", dataset_cache)
        data = torch.load(dataset_cache)
        return data

    dataset_path = f"{path}/{split}.json"
    data = get_dataset_from_file(tokenizer, dataset_path, task_config, max_data, max_block_size=max_block_size)

    if dataset_cache:
        torch.save(data, dataset_cache)

    logger.info("Dataset cached at %s", dataset_cache)

    return data
