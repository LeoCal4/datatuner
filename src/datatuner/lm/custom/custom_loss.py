import logging
import math
import re
from typing import List, Union
import nltk
import torch

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__file__)


def tensor_intersection(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    a_cat_b, counts = torch.cat([a, b]).unique(return_counts=True)
    return a_cat_b[torch.where(counts.gt(1))]


def semantic_fidelity_loss(source_data, target_texts, model_outputs: List[str],
                            missing_data_token_weight: Union[float, torch.Tensor] = 0.5,
                            token_difference_weight: Union[float, torch.Tensor] = 0.5) -> torch.Tensor:
    """_summary_

    Args:
        source_data ()
        target_texts ()
        model_outputs (List[str])
        missing_data_token_weight (Union[float, torch.Tensor], optional): Either a float or a single-item tensor. Defaults to 0.5.
        token_difference_weight (Union[float, torch.Tensor], optional): Either a float or a single-item tensor. Defaults to 0.5.

    Returns:
        torch.Tensor: _description_
    """
    total_missing_data_tokens = []
    total_token_differences = []
    stopwords = set(nltk.corpus.stopwords.words("english"))
    for data, original, generated in zip(source_data, target_texts, model_outputs):
        #* extract single data from complete data, except for yes/no values
        data_tokens = list(re.findall(r"\[([\s\w\d]*)\]", data))
        data_tokens = [token.strip() for token in data_tokens if not token.strip() in ["yes", "no"]]
        #* check if data in target
        missing_data_tokens = len(data_tokens)
        for data_token in data_tokens:
            if data_token in generated:
                missing_data_tokens -= 1
        missing_data_tokens = math.log(missing_data_tokens) if missing_data_tokens != 0 else 0
        total_missing_data_tokens.append(missing_data_tokens)
        #* calculate token difference
        tokenized_original = nltk.tokenize.word_tokenize(original)
        tokenized_generated = nltk.tokenize.word_tokenize(generated)
        #* retain only non-stopwords and non ,/. tokens
        tokenized_original = [token for token in tokenized_original if token not in stopwords and token not in [",", "."]]
        tokenized_original = [token for token in tokenized_generated if token not in stopwords and token not in [",", "."]]
        token_difference = abs(len(tokenized_generated) - len(tokenized_original))
        token_difference = math.log(token_difference) if token_difference != 0 else 0
        total_token_differences.append(token_difference)
    total_missing_data_tokens = torch.tensor(total_missing_data_tokens, dtype=torch.float64)
    total_token_differences = torch.tensor(total_token_differences, dtype=torch.float64)
    sf_loss = missing_data_token_weight * torch.mean(total_missing_data_tokens) + \
                token_difference_weight * torch.mean(total_token_differences)
    return sf_loss
