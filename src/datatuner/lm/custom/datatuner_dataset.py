import csv
import json
import os
import re
import logging
from typing import Dict, List, Tuple

import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers.tokenization_utils import PreTrainedTokenizer


logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__file__)


def remove_non_bracketed_keys(sentence: str) -> str:
    """Removes the keywords which are repeated outside of the angle brackets. 
    For example, the sentence:
        <give_opinion> give opinion ( <name> name: [ SpellForce 3 ], <release_year> release year: [ 2017 ] )
    becomes:
        <give_opinion> ( <name> [ SpellForce 3 ], <release_year> [ 2017 ] )
    
    This assumes that there may be either one or two word (separated by space or _) inside the angle brackets, 
        and that therefore one or two words must be removed afterwards.

    Args:
        sentence (str)

    Returns:
        str
    """
    try:
        tokenized_sentence = sentence.strip().split(" ")
        delete_two = False
        for i, token in enumerate(tokenized_sentence):
            if token[-1] == ">":
                del tokenized_sentence[i+1]
                if delete_two or "_" in token:
                    del tokenized_sentence[i+1]
                delete_two = False
            elif token[0] == "<": # the token is in the form "token>", hence there was another word before it
                delete_two = True
    except:
        print("sentence: ", sentence)
        print("tokenized_sentence: ", tokenized_sentence)
        return ""
    return " ".join(tokenized_sentence)


def get_datatuner_processed_dataset(filename: str, task_config: Dict) -> List[Dict]:
    """Reads the Datatuner-processed dataset from the file specified at filename, using 
    the information contained in task_config.

    Args:
        filename (str)
        task_config (Dict)

    Returns:
        List[Dict]: a list of dicts each containing a "data" and a "text" field for 
            each data point
    """
    #* open the dataset file
    with open(filename, "r") as f:
        data = json.load(f)

    #* get the name of the text fields 
    text_fields = [x for x in task_config["data_shape"] if x["type"] == "text"]
    NEW_TEXT_FIELD_NAMES = ["data", "text"]

    #* iterate over data
    raw_dataset = []
    if "original_data" in task_config:
        original_data_key_name = task_config["original_data"]
        log.info("Adding original data to dataset entries")
    for _, raw_data_point in enumerate(tqdm(data)):
        item = {}
        #* iterate on the text field of the current data point
        #*  the first one is always the data, the second is the sentence 
        for i, text_field in enumerate(text_fields):
            item[NEW_TEXT_FIELD_NAMES[i]] = raw_data_point[text_field["id"]]
        
        if "original_data" in task_config:
            item["original_data"] = raw_data_point[original_data_key_name]

        raw_dataset.append(item)
    return raw_dataset


def get_consistency_dataset(dataset_path: str) -> List[Dict[str, str]]:
    raw_consistency_dataset = []
    with open(dataset_path, "r", encoding="utf-8-sig") as f:
        reader = csv.reader(f, delimiter="|")
        for idx, line in enumerate(reader):
            if idx == 0:
                continue
            item = {}
            item["label"] = line[0]
            item["data"] = line[1]
            item["text"] = line[2]
            raw_consistency_dataset.append(item)
    return raw_consistency_dataset


def read_special_tokens(task_config: Dict, special_tokens_file_path: str) -> List[str]:
    """Read special tokens from file and from the task configuration"""
    tokens = []
    #* Add any special tokens indicated in the file
    with open(special_tokens_file_path, "r") as f:
        tokens += [x.strip() for x in f.readlines() if x.strip()]
    if task_config is not None:
        # add any special tokens defined in the tokenization config
        for item in task_config["data_shape"]:
            if item["type"] == "special":
                tokens += [item["id"]]
        if "extra_special_tokens" in task_config:
            tokens.extend(task_config["extra_special_tokens"])
    #* add base tokens
    tokens += ["<data>", "<text>"]
    return tokens


def process_viggo_key(key: str) -> str:
    key = key.replace("steam", "Steam")
    key = key.replace("mac", "Mac")
    key = key.replace("linux", "Linux")
    key = key.replace("windows", "Windows")
    key = key.replace("esrb", "ESRB")
    key = key.replace("exp", "expected")
    return key


def process_e2e_key(key: str) -> str:
    key = key.replace("eatType", "eat type")
    key = key.replace("priceRange", "price range")
    key = key.replace("familyFriendly", "family friendly")
    key = key.replace("customerRating", "customer rating")
    return key


def process_data(data: str, dataset_name: str) -> Tuple[str, str]:
    """_summary_

    Args:
        data (str): _description_
        dataset_name (str): _description_

    Raises:
        ValueError: _description_

    Returns:
        Tuple[str, str]: return the processed MR/data and its values
    """
    key_value_separator = "="
    slots_separator = "|"
    sentence_separator = "."
    final_sentence = ""
    values = []
    if "webnlg" in dataset_name:
        matches = re.findall(r"(<[\w\s]*>)\s*([^<;]*)(;)?", data)
        for match in matches:
            bracketed_key = match[0]
            key = bracketed_key.strip(" <>")
            value = match[1].strip()
            values.append(value)
            end_sentence = match[2]
            final_token = slots_separator if not end_sentence else end_sentence
            final_sentence += f"{bracketed_key} {key} {key_value_separator} {value} {final_token} "
    elif "viggo" in dataset_name:
        final_sentence = data.split("(")[0].strip() + f" {slots_separator} "
        matches = re.findall(r"(<[\w\s]*>)\s*[\w\s]*:\s*\[\s*([^\]]*)\s*\]", data)
        for match in matches:
            bracketed_key = match[0]
            key = bracketed_key.strip(" <>").replace("_", " ")
            key = process_viggo_key(key)
            value = match[1].strip()
            values.append(value)
            final_sentence += f"{bracketed_key} {key} {key_value_separator} {value} {slots_separator} "
    elif dataset_name == "e2e":
        matches = re.findall(r"(<[\w\s]*>)\s*[\w\s=]*\[\s*([^\]]*)\s*\]", data)
        for match in matches:
            bracketed_key = match[0]
            key = bracketed_key.strip(" <>")
            key = process_e2e_key(key)
            value = match[1].strip()
            values.append(value)
            final_sentence += f"{bracketed_key} {key} {key_value_separator} {value} {slots_separator} "
    elif "jilda" in dataset_name:
        #* jilda's sentences are already well formatted, so we just need to extract the data values
        matches = re.findall(r"[\w\s]*=([^\|\.]*)", data)
        for match in matches:
            values.append(match[0].strip())
        final_sentence = data + "--" # adding 2 dashes since they get removed eventually
    else:
        raise ValueError(f"No configuration for dataset with name {dataset_name}")
    return final_sentence[:-2] + sentence_separator, " | ".join(values) + " |"


class DatatunerDataset(Dataset):
    def __init__(self, raw_dataset: List[Dict], tokenizer: PreTrainedTokenizer, 
            dataset_name: str, dataset_type: str = None,
            data_special_token: str = "data", text_special_token: str = "text",
            text_prefix: str = "from Data to English:",
            max_source_len: int = None, max_target_len: int = None,
            raw_consistency_dataset: List[Dict[str, str]] = None,
            max_consistency_sentences: int = 3) -> None:
        self.raw_dataset = raw_dataset
        self.tokenizer = tokenizer
        self.dataset_name = dataset_name
        self.dataset_type = dataset_type
        self.data_special_token = data_special_token
        self.text_special_token = text_special_token
        self.text_prefix = text_prefix
        self.max_source_len = max_source_len
        self.source_padding_strategy = "max_length" if self.max_source_len else "longest"
        self.max_target_len = max_target_len
        self.target_padding_strategy = "max_length" if self.max_target_len else "longest"
        log.info(f"\tDataset {dataset_name} padded with source: {self.source_padding_strategy} - target: {self.target_padding_strategy}")
        self.processed_sources = []
        self.processed_targets = []
        self.process_raw_dataset()
        self.raw_consistency_dataset = raw_consistency_dataset
        self.max_consistency_sentences = max_consistency_sentences
        self.processed_consistency_sentences = []
        if self.raw_consistency_dataset:
            self.process_raw_consistency_sentences()

    def process_raw_dataset(self):
        """Needs to be implemented by the subclass. 
            Used to populate: 
                - processed_sources
                - processed_targets
                - raw_sources_values """
        raise NotImplementedError

    def process_raw_consistency_sentences(self):
        total_consistency_sentences = []
        current_target_len = len(self.processed_targets.data["input_ids"][0])
        for i, entry in enumerate(self.raw_dataset):
            total_consistency_sentences.append(
                [cons_data["text"] for cons_data in self.raw_consistency_dataset if entry["data"].strip() == cons_data["data"].strip() and cons_data["label"] not in  ["accurate", "repetition"]]
                )
            total_consistency_sentences[i] = total_consistency_sentences[i][:self.max_consistency_sentences]
        for batch in total_consistency_sentences:
            curr_processed_consistency_sentences = self.tokenizer(
                batch, padding="max_length", max_length=current_target_len,
                return_tensors="pt", truncation=True 
            )["input_ids"]
            self.processed_consistency_sentences.append(curr_processed_consistency_sentences)

    def __len__(self) -> int:
        return len(self.raw_dataset)

    def __getitem__(self, idx) -> Tuple[Dict]:
        item = {}
        if "original_data" in self.raw_dataset[idx]:
            item["original_data"] = self.raw_dataset[idx]["original_data"]
        else:
            item["original_data"] = self.raw_dataset[idx]["data"]
        item["processed_data"] = self.raw_dataset[idx]["data"]
        item["target_text"] = self.raw_dataset[idx]["text"]
        if type(item["target_text"]) in (list, tuple):
            item["target_text"] = item["target_text"][-1]
        item["source_input_ids"] = self.processed_sources.data["input_ids"][idx]
        item["source_attention_mask"] = self.processed_sources.data["attention_mask"][idx]
        try:
            item["target_input_ids"] = self.processed_targets.data["input_ids"][idx]
            # item["target_attention_mask"] = self.processed_targets.data["attention_mask"][idx]
        except IndexError:
            item["target_input_ids"] = self.processed_targets[idx]
        item["source_data_values"] = self.raw_sources_values[idx]
        if self.processed_consistency_sentences:
            item["consistency_sentences_input_ids"] = self.processed_consistency_sentences[idx]
        return item


class DatatunerDatasetEncDec(DatatunerDataset):
    def process_raw_dataset(self):
        """Builds the input for the model, processing the source with tokenization + conversion to id + padding.
        Since T5 is an encoder/decoder model, the source contains just the DATA and the targets contains the TEXT.
        Specifically, the source is prepended with the prefix "from data to text" and then the data special token and 
        the text special token are respectively added before and after the DATA.
        As for the targets, they are read from the original Datatuner dataset files, which mostly present more than one sentence,
        hence only the last (hence the right) one is saved.  
        """
        total_sources = []
        total_targets = []
        total_source_values = []
        for i, entry in enumerate(self.raw_dataset):
            # these tokenizer settings are taken from https://huggingface.co/docs/transformers/v4.18.0/en/model_doc/t5#inference
            # self.tokenizer.padding_size = "left"
            # self.tokenizer.padding_token = self.tokenizer.eos_token
            processed_data, values = process_data(entry["data"], self.dataset_name)
            total_source_values.append(values)
            #* substitute with new raw data
            source_string = f"{self.text_prefix} <{self.data_special_token}> {processed_data} <{self.text_special_token}>"
            # source_string = f"{prefix} <{self.data_special_token}> {processed_data} <{self.text_special_token}>"
            self.raw_dataset[i]["data"] = source_string
            total_sources.append(source_string)
            # e2e does not have a list of candidates but just the sentence as a string, so we check for that
            if type(entry["text"]) is list or type(entry["text"]) is tuple:
                target_string = entry['text'][-1]
            else:
                target_string = entry['text']
            total_targets.append(target_string)
        self.processed_sources = self.tokenizer(
            total_sources, padding=self.source_padding_strategy, max_length=self.max_source_len,
            return_tensors="pt", truncation=True
        )
        self.processed_targets = self.tokenizer(
            total_targets, padding=self.target_padding_strategy, max_length=self.max_target_len, 
            return_tensors="pt", truncation=True 
        )
        self.raw_sources_values = total_source_values



class DatatunerDatasetDecOnly(DatatunerDataset):
    def process_raw_dataset(self):
        """In decoder only method, the source must include both the data and the text, while the target
            is composed of a masked part in the beginning (same len as the data) followed by the text.
        """
        total_sources = []
        total_targets = []
        total_source_values = []
        for i, entry in enumerate(self.raw_dataset):
            processed_data, values = process_data(entry["data"], self.dataset_name)
            total_source_values.append(values)
            if "train" in self.dataset_type:
                #* build and tokenize the source string (data token + data + text token + text) ([-1] is needed to take the correct sentence)
                source_string = f"<{self.data_special_token}> {processed_data} <{self.text_special_token}> {entry['text'][-1]}"
                source_string = self.tokenizer(source_string)["input_ids"]
                total_sources.append(source_string[1:]) #! <- this leaves out the starting EOS token from GPT2/OPT, since it will be readded later
                #* tokenize the text part for the target
                if type(entry["text"]) is list or type(entry["text"]) is tuple:
                    only_text_tokens = self.tokenizer(entry["text"][-1])["input_ids"][1:] #! <- this leaves out the starting EOS token from GPT2/OPT, since it will be readded later
                else:
                    only_text_tokens = self.tokenizer(entry["text"])["input_ids"][1:] #! <- this leaves out the starting EOS token from GPT2/OPT, since it will be readded later
                #* manually pad the target string on the left to make it the same size of the source text
                target_string = [
                    self.tokenizer.pad_token_id for _ in range(len(source_string) - len(only_text_tokens))
                    ] + only_text_tokens
                total_targets.append(target_string)
            else:
                #* build and tokenize the complete source string (data token + data + text token + text)
                complete_source_string = f"<{self.data_special_token}> {processed_data} <{self.text_special_token}> {entry['text'][-1]}"
                complete_source_string = self.tokenizer(complete_source_string)["input_ids"]
                #* build and tokenize the validation source string (data token + data + text token)
                val_source_string = f"<{self.data_special_token}> {processed_data} <{self.text_special_token}>"
                val_source_string = self.tokenizer(val_source_string)["input_ids"]
                #* manually pad the source string on the right to make it the same size of the complete source text
                val_source_string = val_source_string + [
                    self.tokenizer.pad_token_id for _ in range(len(complete_source_string) - len(val_source_string))
                    ] # TODO NEED TO UPDATE ATTENTION MASKS TOO
                total_sources.append(val_source_string[1:]) #! <- this leaves out the starting EOS token from GPT2/OPT, since it will be readded later
                #* tokenize the text part for the target
                if type(entry["text"]) is list or type(entry["text"]) is tuple:
                    only_text_tokens = self.tokenizer(entry["text"][-1])["input_ids"][1:] #! <- this leaves out the starting EOS token from GPT2/OPT, since it will be readded later
                else:
                    only_text_tokens = self.tokenizer(entry["text"])["input_ids"][1:] #! <- this leaves out the starting EOS token from GPT2/OPT, since it will be readded later
                #* manually pad the target string on the left to make it the same size of the source text
                target_string = [
                    self.tokenizer.pad_token_id for _ in range(len(complete_source_string) - len(only_text_tokens))
                    ] + only_text_tokens
                total_targets.append(target_string)
        self.processed_sources = self.tokenizer.batch_encode_plus(
            total_sources, padding="longest", 
            return_tensors="pt", is_split_into_words=True
        )
        target_max_length = max([len(entry) for entry in total_sources]) + 1
        padded_targets = []
        # TODO change back to tokenizer + remove \s at the start
        for target in total_targets:
            target = target + [self.tokenizer.pad_token_id for _ in range(target_max_length - len(target))] 
            padded_targets.append(target)
        self.processed_targets = torch.tensor(padded_targets)
        # log.info(f"processed targets: {self.processed_targets[0]}")
        # self.processed_targets = self.tokenizer.batch_encode_plus(
        #     total_targets, padding="max_length", max_length=target_max_length,
        #     return_tensors="pt", truncation=True, is_split_into_words=True
        # )
        self.raw_sources_values = total_source_values



def get_data_loaders(base_dataset_path: str, task_config: Dict, tokenizer: PreTrainedTokenizer,
        text_prefix: str = "", model_type: str = "enc_dec",
        dataset_types: List[str] = ["train", "validation"], 
        batch_sizes: Dict[str, int] = {"train": 8, "validation": 8},
        max_source_len: int = None, max_target_len: int = None,
        consistency_dataset_path: str = None) -> Tuple[DataLoader]:
    assert dataset_types == list(batch_sizes.keys())
    data_loaders = {}
    #! assuming that the name of the dataset folder is the dataset name
    dataset_name = base_dataset_path.split(os.sep)[-1]
    for dataset_type in dataset_types:
        current_dataset_path = os.path.join(base_dataset_path, f"{dataset_type}.json")
        raw_dataset = get_datatuner_processed_dataset(current_dataset_path, task_config)
        raw_consistency_dataset = None
        if consistency_dataset_path:
            current_consistency_dataset_path = os.path.join(consistency_dataset_path, f"{dataset_type}.tsv")
            raw_consistency_dataset = get_consistency_dataset(current_consistency_dataset_path)
        if model_type == "enc_dec":
            current_dataset = DatatunerDatasetEncDec(raw_dataset, tokenizer, dataset_name,
                text_prefix=text_prefix,
                max_source_len=max_source_len, max_target_len=max_target_len, 
                raw_consistency_dataset=raw_consistency_dataset)
        elif model_type == "dec_only":
            current_dataset = DatatunerDatasetDecOnly(raw_dataset, tokenizer, dataset_name, dataset_type=dataset_type,
                text_prefix=text_prefix,
                max_source_len=max_source_len, max_target_len=max_target_len, 
                raw_consistency_dataset=raw_consistency_dataset)
        else:
            raise ValueError(f"Unknown model type {model_type}")
        log.info(f"Source data: {current_dataset[0]['processed_data']}")
        log.info(f"Source data values: {current_dataset[0]['source_data_values']}")
        log.info(f"Target text: {current_dataset[0]['target_text']}")
        log.info(f"Source input ids: {current_dataset[0]['source_input_ids'].shape} - {current_dataset[0]['source_input_ids']}")
        log.info(f"Source input ids decoded: {tokenizer.decode(current_dataset[0]['source_input_ids'])}")
        log.info(f"Target input ids: {current_dataset[0]['target_input_ids'].shape} - {current_dataset[0]['target_input_ids']}")
        log.info(f"Target input ids decoded: {tokenizer.decode(current_dataset[0]['target_input_ids'])}")
        batch_size = batch_sizes[dataset_type]
        data_loaders[dataset_type] = DataLoader(
            current_dataset, batch_size=batch_size, shuffle=bool(dataset_type=="train"))
    return tuple(data_loaders.values()) if len(data_loaders) > 1 else tuple(data_loaders.values())[0]
