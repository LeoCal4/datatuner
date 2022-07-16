import logging
import math
import re
from typing import List, Union

import utils
ser_calculator = utils.import_ser_calculator()


import torch
import torch.nn.functional as F
from transformers.tokenization_utils import PreTrainedTokenizer



logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__file__)


def differentiable_tensor_intersection(data: torch.Tensor, predicted: torch.Tensor) -> torch.Tensor:
    """Differentiable tensor intersection. Searches the elements of predicted in data,
    then returns a sliced version of predicted, containing only the elements in common.

    Args:
        data (torch.Tensor)
        predicted (torch.Tensor)

    Returns:
        torch.Tensor: tensor containing the intersection of data and predicted
    """
    base_mask = torch.zeros_like(predicted).bool()
    checked_tokens = []
    for token in data:
        if token in checked_tokens:
            continue
        checked_tokens.append(token)
        base_mask += predicted.eq(token)
    return predicted[base_mask]


def tensor_intersection(a, b) -> torch.Tensor:
    """Returns a tensor with the elements contained in both a and b"""
    a_cat_b, counts = torch.cat([a[a > 0], b[b > 0]]).unique(return_counts=True)
    return a_cat_b[counts.gt(1)]


def differentiable_tensor_len(a: torch.Tensor) -> torch.Tensor:
    """Calculates the length (hence first dimension) of the tensor in a differentiable way.
    Only non-zero elements are considered, since 0 is the padding token id by default, otherwise all the input data
    would have the same length.

    Args:
        a (torch.Tensor)

    Returns:
        torch.Tensor:
    """
    nonzero_a = a[a.nonzero()]
    return torch.sum((nonzero_a)/(nonzero_a))


def soft_argmax(a: torch.Tensor) -> torch.Tensor:
    """A differentiable way of computing the argmax.
    Assumes tensor of shape [batch_size, sequence_length, vocab_size]

    https://stackoverflow.com/questions/46926809/getting-around-tf-argmax-which-is-not-differentiable
    https://bouthilx.wordpress.com/2013/04/21/a-soft-argmax/

    Args:
        a (torch.Tensor): _description_

    Returns:
        torch.Tensor: _description_
    """
    a_range = torch.arange(a.size(-1)).to(a.device)
    return torch.sum(torch.softmax(a*1e10, -1)*a_range, -1)


def differentiable_semantic_fidelity_loss(source_data_ids, target_texts_ids, logits: List[str],
                            missing_data_token_weight: Union[float, torch.Tensor] = 0.5,
                            token_difference_weight: Union[float, torch.Tensor] = 0.5) -> torch.Tensor:
    """A (theorically) differentiable implementation of the semantic fidelity loss.
    The loss comprises two elements. For each triple (input data - input text - output text) we calculate
        - the absolute difference between the number of data tokens and the number of data tokens
            actually used in the predicted sentence (this is technically an approximation of said value,
            given the implementation of tensor_intersection)
        - the absolute difference between the input text and the output text lengths'
    Both of those values are aggregated after a log is applied to them, to keep the same order of magnitude
    as the cross entropy loss.

    The final loss is calculated as a weighted average of said values.

    Args:
        source_data_ids ()
        target_texts_ids ()
        logits (): model outputs
        missing_data_token_weight (Union[float, torch.Tensor], optional): Either a float or a single-item tensor. Defaults to 0.5.
        token_difference_weight (Union[float, torch.Tensor], optional): Either a float or a single-item tensor. Defaults to 0.5.

    Returns:
        torch.Tensor: single-item tensor containing the calculated loss
    """
    total_missing_data_tokens = []
    total_token_differences = []
    predictions = soft_argmax(logits).floor() # cannot simply convert to int without losing grad
    for data, target, predicted in zip(source_data_ids, target_texts_ids, predictions):
        #* check how many data tokens are found in predicted
        data = data.float()
        source_intersect_predicted = differentiable_tensor_intersection(data, predicted)
        intersection_len = differentiable_tensor_len(source_intersect_predicted)
        data_len = differentiable_tensor_len(data)
        missing_data_tokens = torch.abs(data_len - intersection_len)
        total_missing_data_tokens.append(missing_data_tokens)
        #* calculate token difference
        target_len = differentiable_tensor_len(target)
        predicted_len = differentiable_tensor_len(predicted)
        token_difference_length = torch.abs(target_len - predicted_len)
        total_token_differences.append(token_difference_length)
    total_missing_data_tokens = torch.stack(total_missing_data_tokens)
    total_token_differences = torch.stack(total_token_differences)
    sf_loss = missing_data_token_weight * torch.log(torch.mean(total_missing_data_tokens)) + \
                token_difference_weight * torch.log(torch.mean(total_token_differences))
    return sf_loss


def semantic_fidelity_loss_with_confidences(source_data_ids: torch.Tensor, logits: torch.Tensor) -> torch.Tensor:
    """Loss based on the product of the means two elements: 
        - ratio between the length of the intersection between generated/predicted 
        sentences with the data and the length of the data (all in tensors/ids form)
        - confidences

    Returns:
        torch.Tensor
    """
    total_len_ratios = []
    softmaxes = F.softmax(logits, dim=1)
    confidences, predictions = torch.max(softmaxes, -1)
    for data, predicted in zip(source_data_ids, predictions):
        source_intersect_predicted = tensor_intersection(data, predicted)
        len_ratio = source_intersect_predicted.size(0)/data.size(0)
        total_len_ratios.append(len_ratio)
    mean_conf = torch.mean(confidences)
    mean_ratios = sum(total_len_ratios) / len(total_len_ratios)
    sf_loss = mean_conf * mean_ratios
    return sf_loss


def word_based_semantic_fidelity_loss(
    source_data_values: List[str], 
    target_data: List[str], 
    logits: torch.Tensor, 
    tokenizer: PreTrainedTokenizer) -> torch.Tensor:
    """SM loss which calculates the following:
        log 1/N SUM_i | intersection(tags, target) | / | intersection(tags, predicted) |
    using the natural language version tags, target and predicted to get their intersection.

    Returns:
        torch.Tensor
    """
    total_len_ratios = []
    softmaxes = F.softmax(logits, dim=1)
    _, predictions = torch.max(softmaxes, -1)
    batch_sentences = tokenizer.batch_decode(predictions, skip_special_tokens=True)
    for data, target, predicted in zip(source_data_values, target_data, batch_sentences):
        #* get all the data values tokens, separating both different slots and slots where more values are separated by a comma
        data_tokens = list(re.findall(r"\s?([^|,]*)[|,]", data))
        #* strip tokens and avoid getting "yes" and "no", as they don't provide anything
        data_tokens = [token.strip() for token in data_tokens if token.strip() not in ["yes", "no"]]
        data_tokens_in_target = 0.1 # avoids 0 division
        for data_token in data_tokens:
            data_tokens_in_target += int(data_token in target)
        data_tokens_in_pred = 0.1 # avoids 0 division
        for data_token in data_tokens:
            data_tokens_in_pred += int(data_token in predicted)
        len_ratio = max(data_tokens_in_target / data_tokens_in_pred, 1.0)
        total_len_ratios.append(len_ratio)
    mean_ratios = sum(total_len_ratios) / len(total_len_ratios)
    return math.log(mean_ratios)


def word_based_semantic_fidelity_loss_with_confidences(
    source_data_values: List[str], 
    target_data: List[str], 
    logits: torch.Tensor, 
    tokenizer: PreTrainedTokenizer) -> torch.Tensor:
    """SM loss which calculates the following:
        log 1/N SUM_i (conf_i * | intersection(tags, target) | / | intersection(tags, predicted) |)
    using the natural language version of tags, target and predicted to get their intersection.

    Returns:
        torch.Tensor
    """
    total_len_ratios = []
    softmaxes = F.softmax(logits, dim=1)
    confidences, predictions = torch.max(softmaxes, -1)
    batch_sentences = tokenizer.batch_decode(predictions, skip_special_tokens=True)
    for data, target, predicted in zip(source_data_values, target_data, batch_sentences):
        #* get all the data values tokens, separating both different slots and slots where more values are separated by a comma
        data_tokens = list(re.findall(r"\s?([^|,]*)[|,]", data))
        #* strip tokens and avoid getting "yes" and "no", as they don't provide anything
        data_tokens = [token.strip() for token in data_tokens if token.strip() not in ["yes", "no"]]
        data_tokens_in_target = 0.1 # avoids 0 division
        for data_token in data_tokens:
            data_tokens_in_target += int(data_token in target)
        data_tokens_in_pred = 0.1 # avoids 0 division
        for data_token in data_tokens:
            data_tokens_in_pred += int(data_token in predicted)
        len_ratio = max(data_tokens_in_target / data_tokens_in_pred, 1.0)
        total_len_ratios.append(len_ratio)
    mean_conf = torch.mean(confidences)
    mean_ratios = sum(total_len_ratios) / len(total_len_ratios)
    sf_loss_pre_log = mean_conf * mean_ratios
    # sf_loss = torch.log(sf_loss_pre_log)
    # log.info(f"SF loss: {mean_conf} * {mean_ratios} = {sf_loss_pre_log} => {sf_loss}")
    return sf_loss_pre_log


def calculate_dcs(logits: torch.Tensor, original_data: List[str], tokenizer: PreTrainedTokenizer, dataset_name: str) -> torch.Tensor:
    softmaxes = F.softmax(logits, dim=1)
    confidences, predictions = torch.max(softmaxes, -1)
    batch_sentences = tokenizer.batch_decode(predictions, skip_special_tokens=True)
    ser_value, _ = ser_calculator.calculate_ser(original_data, batch_sentences, dataset_name)
    mean_conf = torch.mean(confidences)
    return torch.abs((1.0-ser_value) - mean_conf)
