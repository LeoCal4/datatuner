import torch
import re
import logging 

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__file__)

def semantic_fidelity_loss(source_data, model_outputs, 
                            missing_weight: float = 0.5, ratio_weight: float = 0.5) -> torch.tensor:
    total_losses = []
    for data, generated in zip(source_data, model_outputs):
        #* extract single data from complete data
        data_tokens = list(re.findall(r"\[([\s\w\d]*)\]", data))
        data_tokens = [token.strip() for token in data_tokens]
        #* check if data in target
        missing_data_tokens = len(data_tokens)
        for data_token in data_tokens:
            if data_token in generated:
                missing_data_tokens -= 1
        #* calculate token ratio
        if len(data_tokens) == 0:
            tokens_ratio = 1
        else:
            tokens_ratio = len(generated.split(" ")) / len(data_tokens)
        # log.info(f"Missing: {missing_data_tokens} - Ratio: {tokens_ratio}")
        total_losses.append(
            missing_weight * missing_data_tokens + ratio_weight * tokens_ratio
        )
    #? loss for each batch or total average or total sum
    total_loss = sum(total_losses) / len(total_losses)
    return torch.tensor([total_loss], dtype=torch.float64)