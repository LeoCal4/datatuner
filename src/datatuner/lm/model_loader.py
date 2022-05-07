import json
import logging
from pathlib import Path
from pyexpat import model

import mlflow
from datatuner.lm import custom_tokenizer
from datatuner.lm.custom_gpt2 import custom_gpt2_with_smoothing, custom_gpt2_with_agumented_attention, custom_t5_with_agumented_attention
from datatuner.lm.data_loader import PAD_TOKEN
from transformers import (
    GPT2DoubleHeadsModel,
    GPT2LMHeadModel,
    GPT2Tokenizer,
    OpenAIGPTDoubleHeadsModel,
    OpenAIGPTLMHeadModel,
    OpenAIGPTTokenizer,
)
try:
    from transformers import T5Tokenizer
except:
    pass
try:
    from transformers import GPT2_PRETRAINED_MODEL_ARCHIVE_MAP
except ImportError:
    from transformers.models.gpt2.modeling_gpt2 import GPT2_PRETRAINED_MODEL_ARCHIVE_LIST
    GPT2_PRETRAINED_MODEL_ARCHIVE_MAP = GPT2_PRETRAINED_MODEL_ARCHIVE_LIST

from transformers.tokenization_utils import PreTrainedTokenizer

logger = logging.getLogger(__file__)


def load_pretrained_tokenizer(model_checkpoint, model_type):
    """Load pretrained tokenizer"""
    tokenizer_class = OpenAIGPTTokenizer if "openai-gpt" in model_type else GPT2Tokenizer
    if "openai-gpt" in model_type:
        tokenizer_class = OpenAIGPTTokenizer
    elif "gpt2" in model_type:
        tokenizer_class = GPT2Tokenizer
    elif "t5" in model_type:
        tokenizer_class = T5Tokenizer
    tokenizer = tokenizer_class.from_pretrained(model_checkpoint)
    if "t5" not in model_type:
        PreTrainedTokenizer.tokenize = custom_tokenizer.tokenize
    return tokenizer


def load_training_args(run_id):
    client = mlflow.tracking.MlflowClient()
    training_args_file = client.download_artifacts(run_id, "training/model_training_args.json")
    model_training_args = json.load(open(training_args_file))
    return model_training_args


def get_model_directory(model_checkpoint=None):
    """Get the model directory; if `model_checkpoint` is a folder, it is returned;
     if it is a shortcut name for a Hugging Face model, the name is returned for handling downstream;
     if it's a run_id, the folder is obtained from mlflow."""
    is_local = True
    if Path(model_checkpoint).exists():
        return Path(model_checkpoint), is_local
    elif (type(GPT2_PRETRAINED_MODEL_ARCHIVE_MAP) is dict and model_checkpoint in GPT2_PRETRAINED_MODEL_ARCHIVE_MAP.keys()) or \
        (type(GPT2_PRETRAINED_MODEL_ARCHIVE_MAP) is list and model_checkpoint in GPT2_PRETRAINED_MODEL_ARCHIVE_MAP) or \
            model_checkpoint == "t5-base": # TODO horrible
        is_local = False
        return model_checkpoint, is_local
    else:
        client = mlflow.tracking.MlflowClient()
        run_info = client.get_run(model_checkpoint)
        is_local = False
        return Path(run_info.info.artifact_uri) / "training", is_local


def read_special_tokens(task_config=None, special_tokens_file=None, dataset_path=None):
    """Read special tokens from file and from the task configuration"""
    tokens = []
    # If no special tokens file is explicitly passed, we try finding a special_tokens.txt file in the model directory
    if special_tokens_file is None:
        if dataset_path is not None:
            special_tokens_file = Path(dataset_path) / "special_tokens.txt"

    # Add any special tokens indicated in the file
    if special_tokens_file is not None and special_tokens_file.exists():
        tokens += [x for x in special_tokens_file.read_text().split("\n") if x.strip()]
        logger.info(f"read {len(tokens)} special tokens from {special_tokens_file}")

    if task_config is not None:
        # add any special tokens defined in the tokenization
        for item in task_config["data_shape"]:
            if item["type"] == "special":
                tokens += [item["id"]]

        if "extra_special_tokens" in task_config:
            tokens.extend(task_config["extra_special_tokens"])

    # Add basic eos and padding tokens
    tokens += [PAD_TOKEN, "<eos>"]

    return tokens


def load_pretrained(
        model_directory,
        model_type=None,
        smoothing=0.0,
        output_attentions=True,
        output_hidden_states=True,
        multitask=False,
        special_tokens_file=None,
        task_config=None,
        dataset_path=None,
        use_custom_t5=False,
        attention_in_last_layer=False,
        mask_attention_in_last_layer=False,
        sum_attention_to_output=False,
        normalize_lm_output=False,
        normalize_attention_sum=False,
        **kwargs,
):
    """Load pretrained model"""
    print("Get pretrained model and tokenizer")
    model_directory = str(model_directory)

    try:
        model_training_args = json.load(open(Path(model_directory) / "model_training_args.json"))
        if "gpt2" in model_training_args["model_directory"]:
            model_type = "gpt2"
        elif "openai-gpt" in model_training_args["model_directory"]:
            model_type = "openai-gpt"

        multitask = model_training_args["multitask"]
    except:
        pass

    if model_type is None:
        model_type = model_directory

    tokenizer = load_pretrained_tokenizer(model_directory, model_type)

    if smoothing > 0:
        model_class = custom_gpt2_with_smoothing(smoothing=smoothing)
    elif attention_in_last_layer:
        model_class = custom_gpt2_with_agumented_attention(
            normalize_lm_output=normalize_lm_output, normalize_attention_sum=normalize_attention_sum,
            )
    elif use_custom_t5:
        model_class = custom_t5_with_agumented_attention( 
            normalize_lm_output=normalize_lm_output, normalize_attention_sum=normalize_attention_sum,
            )
        model_type = "t5-base"
    elif "gpt2" in model_type:
        if multitask:
            model_class = GPT2DoubleHeadsModel
        else:
            model_class = GPT2LMHeadModel
    elif "openai-gpt" in model_type:
        if multitask:
            model_class = OpenAIGPTDoubleHeadsModel
        else:
            model_class = OpenAIGPTLMHeadModel
    else:
        raise ValueError(
            "Invalid model type; make sure to pass the actual model_type if your checkpoint name or model name does not have the model type in them"
        )

    model = model_class.from_pretrained(
        model_directory, output_attentions=output_attentions, output_hidden_states=output_hidden_states, **kwargs
    )

    return model, tokenizer
