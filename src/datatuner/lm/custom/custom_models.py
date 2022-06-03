from typing import Dict
import logging

import torch
import torch.nn as nn

from datatuner.lm.custom import custom_loss


logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__file__)


class CustomT5Model(nn.Module):
    def __init__(self, model, tokenizer, device="cuda", consistency_loss_weight: float = 0.1) -> None:
        super(CustomT5Model, self).__init__()
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.consistency_loss_weight = consistency_loss_weight

    def _inner_forward(self, batch: Dict[str, torch.Tensor], pad_token_id: int = 0):
        """When we call model() with labels, they will be:
            - automatically right shifted by 1 (for teacher forcing)
            - prepended by BOS=Beginning of sequence which is a PAD token
            - any token that was -100 will be masked_fill_ to <pad> for teacher forcing

        Args:
            device (str)
            batch (_type_)
            pad_token_id (int): defaults to 0

        Returns:
            float: loss value
        """
        source_ids = batch["source_input_ids"].to(self.device, dtype=torch.long)
        source_mask = batch["source_attention_mask"].to(self.device, dtype=torch.long)
        target_ids = batch["target_input_ids"].to(self.device, dtype=torch.long)
        #* padded ids are set to -100, so that they are ignored during loss calculation
        target_ids[target_ids[: ,:] == pad_token_id] = -100
        label_ids = target_ids.to(self.device)
        out_dict = self.model(source_ids, attention_mask=source_mask, labels=label_ids, return_dict=True)
        loss = out_dict[0]
        logits = out_dict[1]
        return loss, logits

    def consistency_loss(self, logits_batches: torch.Tensor, 
                            consistency_sentences_input_ids_batches: torch.Tensor, 
                            labels_batches: torch.Tensor) -> torch.Tensor:
        loss_func = nn.CrossEntropyLoss(ignore_index=-100)
        total_consistency_loss = None
        for logits, consistency_sentences_ids, labels_batches in zip(logits_batches, consistency_sentences_input_ids_batches, labels_batches):
            curr_consistency_loss = None
            for consistency_sentence_ids in consistency_sentences_ids:
                # save the losses
                if curr_consistency_loss is None:
                    curr_consistency_loss = loss_func(logits, consistency_sentence_ids)
                else:
                    curr_consistency_loss = torch.vstack((curr_consistency_loss, loss_func(logits, consistency_sentence_ids)))
            # average them
            curr_consistency_loss = torch.mean(curr_consistency_loss)
            positive_loss = loss_func(logits, labels_batches)
            current_total_loss = positive_loss - self.consistency_loss_weight * curr_consistency_loss
            if total_consistency_loss is None:
                total_consistency_loss = current_total_loss
            else:
                total_consistency_loss = torch.vstack((total_consistency_loss, current_total_loss))
        return torch.mean(total_consistency_loss)


    def forward(self, batch: Dict[str, torch.Tensor]):
        loss, logits = self._inner_forward(batch)
        # if "consistency_sentences_input_ids" in batch:
        #     loss = self.consistency_loss(
        #         logits, 
        #         batch["consistency_sentences_input_ids"].to(device=self.device, dtype=torch.long),
        #         batch["target_input_ids"].to(device=self.device, dtype=torch.long)
        #     )
        # else:
        #     sm_loss = custom_loss.semantic_fidelity_loss(
        #         batch["source_input_ids"].to(device=self.device, dtype=torch.long),
        #         batch["target_input_ids"].to(device=self.device, dtype=torch.long),
        #         logits)
            # log.info("sm_loss: ", sm_loss.item())
            # loss = loss + 0.5 * sm_loss 
        return loss
    
    def inference(self, batch: Dict[str, torch.Tensor]):
        source_ids = batch["source_input_ids"].to(self.device, dtype=torch.long)
        source_mask = batch["source_attention_mask"].to(self.device, dtype=torch.long)
        return self.model.generate(
            source_ids, 
            attention_mask=source_mask, 
            max_length=200,
            num_beams=5,
            early_stopping=True,
            num_return_sequences=5,
        ) #! hardcoded length TODO
