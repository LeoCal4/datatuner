import os
from typing import Dict, List, Tuple
import json
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from transformers.tokenization_utils import PreTrainedTokenizer
from transformers import T5Tokenizer


def get_raw_dataset(filename: str, task_config: Dict) -> List[Dict]:
    """Reads the dataset from the file specified at filename, using 
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
    for _, raw_data_point in enumerate(tqdm(data)):
        item = {}
        #* iterate on the text field of the current data point
        #*  the first one is always the data, the second is the sentence 
        for i, text_field in enumerate(text_fields):
            item[NEW_TEXT_FIELD_NAMES[i]] = raw_data_point[text_field["id"]]

        #* add any needed extra_field
        if "extra_fields" in task_config: # found in webnlg
            for extra_field in task_config["extra_fields"]:
                item[extra_field] = raw_data_point[extra_field]
        
        raw_dataset.append(item)
    return raw_dataset


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
    print(f"read {len(tokens)} special tokens from {special_tokens_file_path} and base tokens")
    #? add mask token?
    return tokens


class DatatunerDataset(Dataset):
    def __init__(self, raw_dataset: List[Dict], tokenizer: PreTrainedTokenizer, is_validation: bool=False,
            data_special_token: str = "data", text_special_token: str = "text",
            max_source_len: int = None, max_target_len: int = None) -> None:
        self.raw_dataset = raw_dataset
        self.tokenizer = tokenizer
        self.is_validation = True
        # self.is_validation = is_validation
        self.data_special_token = data_special_token
        self.text_special_token = text_special_token
        self.max_source_len = max_source_len
        self.source_padding_strategy = "max_length" if self.max_source_len else "longest"
        self.max_target_len = max_target_len #! unused for now
        self.processed_sources = []
        self.processed_targets = []
        self.apply_tokenizer_to_raw_dataset()

    def apply_tokenizer_to_raw_dataset(self):
        """Builds the input for the model, normally processing the source (tokenization + conversion to id + padding)
            and building the target as such:
            - tokenization
            - pad the text to the left to match the len of the related source
                - SOURCE: <data> DATA <text> TEXT
                - TARGET:  PAD   PAD   PAD   TEXT
              where DATA, TEXT and PAD possibly represent more than one token. 
            - pad normally to the right together with the source
        In case the dataset is used for validation/testing, the TEXT part is omitted from the source.
        """
        total_data = []
        total_targets = []
        for entry in self.raw_dataset:
            if not self.is_validation:
                self.tokenizer.padding_size = "right"
                #* build and tokenize the source string (data token + data + text token + text) ([-1] is needed to take the correct sentence)
                source_string = f"<{self.data_special_token}> {entry['data']} <{self.text_special_token}> {entry['text'][-1]}"
                source_string = self.tokenizer(source_string)["input_ids"]
                total_data.append(source_string)
                #* tokenize the text part for the target
                only_text_tokens = self.tokenizer(entry["text"][-1])["input_ids"]
                #* manually pad the target string on the left to make it the same size of the source text
                target_string = [
                    self.tokenizer.pad_token_id for _ in range(len(source_string) - len(only_text_tokens))
                    ] + only_text_tokens
                total_targets.append(target_string)
            else: #* during validation we want to avoid adding the text part of the input, the rest is the same
                # these tokenizer settings are taken from https://huggingface.co/docs/transformers/v4.18.0/en/model_doc/t5#inference
                # self.tokenizer.padding_size = "left"
                # self.tokenizer.padding_token = self.tokenizer.eos_token
                #* build and tokenize the complete source string (data token + data + text token + text)
                complete_source_string = f"<{self.data_special_token}> {entry['data']} <{self.text_special_token}> {entry['text'][-1]}"
                complete_source_string = self.tokenizer(complete_source_string)["input_ids"]
                #* build and tokenize the validation source string (data token + data + text token)
                val_source_string = f"<{self.data_special_token}> {entry['data']} <{self.text_special_token}>"
                val_source_string = self.tokenizer(val_source_string)["input_ids"][:-1] #! <- this leaves out the EOS token
                #* manually pad the source string on the right to make it the same size of the complete source text
                val_source_string = val_source_string + [
                    self.tokenizer.pad_token_id for _ in range(len(complete_source_string) - len(val_source_string))
                    ] # TODO NEED TO UPDATE ATTENTION MASKS TOO
                total_data.append(val_source_string)
                #* tokenize the text part for the target
                only_text_tokens = self.tokenizer(entry["text"][-1])["input_ids"]
                #* manually pad the target string on the left to make it the same size of the source text
                target_string = [
                    self.tokenizer.pad_token_id for _ in range(len(complete_source_string) - len(only_text_tokens))
                    ] + only_text_tokens
                total_targets.append(target_string)

        #* finish using batch_encode_plus, since __call__ does not accept input_ids
        self.processed_sources = self.tokenizer.batch_encode_plus(
            total_data, padding=self.source_padding_strategy, return_tensors="pt", is_split_into_words=True, truncation=True, 
            max_length=self.max_source_len
        )
        #* pad labels to the same len of the source 
        #! change the pad token to the mask one? LET'S TO THIS N rip
        target_len = len(self.processed_sources["input_ids"][0])
        self.processed_targets = self.tokenizer.batch_encode_plus(
            total_targets, padding="max_length", max_length=target_len, return_tensors="pt", is_split_into_words=True
        )

    def __len__(self) -> int:
        return len(self.raw_dataset)

    def __getitem__(self, idx) -> Tuple[Dict]:
        item = {}
        item["source_input_ids"] = self.processed_sources.data["input_ids"][idx]
        item["source_attention_mask"] = self.processed_sources.data["attention_mask"][idx]
        item["target_input_ids"] = self.processed_targets.data["input_ids"][idx]
        item["target_attention_mask"] = self.processed_targets.data["attention_mask"][idx]
        return item


def get_data_loaders(base_dataset_path: str, task_config: Dict, tokenizer: PreTrainedTokenizer,
        train_batch_size: int = 8, val_batch_size: int = 1,
        max_source_len: int = None, max_target_len: int = None) -> Dict[str, DataLoader]:
    data_loaders = {}
    for dataset_type in ["train", "validation"]:
        current_dataset_path = os.path.join(base_dataset_path, f"{dataset_type}.json")
        raw_dataset = get_raw_dataset(current_dataset_path, task_config)
        current_dataset = DatatunerDataset(raw_dataset, tokenizer, is_validation=bool(dataset_type=="validation"),
            max_source_len=max_source_len, max_target_len=max_target_len)
        batch_size = train_batch_size if dataset_type == "train" else val_batch_size
        data_loaders[dataset_type] = DataLoader(
            current_dataset, batch_size=batch_size, shuffle=bool(dataset_type=="train"))
    return data_loaders["train"], data_loaders["validation"]


if __name__ == "__main__":
    base_dataset_path = r"/content/drive/MyDrive/Tesi/datasets/webnlg/validation.json"
    task_config_path = r"/content/drive/MyDrive/Tesi/datasets/webnlg/webnlg_task_config.json"
    task_config = json.load(open(task_config_path, "r"))
    special_tokens_path = r"/content/drive/MyDrive/Tesi/datasets/webnlg/special_tokens.txt"
    special_tokens = read_special_tokens(task_config, special_tokens_path)
    tokenizer = T5Tokenizer.from_pretrained("t5-small")
    tokenizer.add_tokens(special_tokens)
    data_loaders = get_data_loaders(base_dataset_path, task_config, tokenizer)
