import logging
from typing import List, Union
import torch

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__file__)


def tensor_intersection(data: torch.Tensor, predicted: torch.Tensor) -> torch.Tensor:
    """Differentiable tensor intersection. Searches the elements of predicted in data,
    then returns a sliced version of predicted, containing only the elements in common.

    Args:
        data (torch.Tensor)
        predicted (torch.Tensor)

    Returns:
        torch.Tensor: tensor containing the intersection of data and predicted
    """
    # a_cat_b, counts = torch.cat([a, b]).unique(return_counts=True)
    # return a_cat_b[counts.gt(1)] # counts.gt returns a bool mask
    # return a_cat_b[torch.where(counts.gt(1))] # where() returns the indexes of True elements
    base_mask = torch.zeros_like(predicted).bool()
    checked_tokens = []
    for token in data:
        if token in checked_tokens:
            continue
        checked_tokens.append(token)
        base_mask += predicted.eq(token)
    return predicted[base_mask]


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


def semantic_fidelity_loss(source_data_ids, target_texts_ids, logits: List[str],
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
        source_intersect_predicted = tensor_intersection(data, predicted)
        intersection_len = differentiable_tensor_len(source_intersect_predicted)
        data_len = differentiable_tensor_len(data)
        log_missing_data_tokens = torch.abs(data_len - intersection_len)
        total_missing_data_tokens.append(log_missing_data_tokens)
        #* calculate token difference
        target_len = differentiable_tensor_len(target)
        predicted_len = differentiable_tensor_len(predicted)
        log_token_difference = torch.abs(target_len - predicted_len)
        total_token_differences.append(log_token_difference)
    total_missing_data_tokens = torch.stack(total_missing_data_tokens)
    total_token_differences = torch.stack(total_token_differences)
    sf_loss = missing_data_token_weight * torch.log(torch.mean(total_missing_data_tokens)) + \
                token_difference_weight * torch.log(torch.mean(total_token_differences))
    return sf_loss
