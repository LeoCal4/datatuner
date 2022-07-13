from typing import Dict
import logging

import torch
import torch.nn as nn

from datatuner.lm.custom import custom_loss


logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__file__)


class CustomT5Model(nn.Module):
    def __init__(self, model, tokenizer, device="cuda", current_dataset: str ="",
        use_consistency_loss: bool = False, consistency_loss_weight: float = 0.1, 
        use_sf_loss: bool = False, sf_loss_alpha: float = 0.5,
        use_dcs_loss: bool = False, dcs_beta: float = 1.0) -> None:
        super(CustomT5Model, self).__init__()
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.use_consistency_loss = use_consistency_loss
        self.current_dataset = current_dataset
        if self.use_consistency_loss:
            log.info(f"\n\tUsing consistency loss")
            self.consistency_loss_weight = consistency_loss_weight
        self.use_sf_loss = use_sf_loss
        if self.use_sf_loss:
            log.info(f"\n\tUsing semantic fidelity loss with confidences")
            self.sf_loss_alpha = sf_loss_alpha
            log.info(f"\n\tLoss: (1-{self.sf_loss_alpha})*CE + {self.sf_loss_alpha}*SF")
        self.use_dcs_loss = use_dcs_loss
        if self.use_dcs_loss:
            log.info(f"\n\tUsing DCS agumented loss with beta = {dcs_beta}")
            self.dcs_beta = dcs_beta
            log.info(f"\n\tLoss: CE + {self.dcs_beta}*DCS")


    def _inner_forward(self, batch: Dict[str, torch.Tensor]):
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
        target_ids[target_ids[:, :] == self.tokenizer.pad_token_id] = -100
        label_ids = target_ids.to(self.device)
        # out_dict = self.model(source_ids, labels=label_ids, return_dict=True)
        out_dict = self.model(source_ids, attention_mask=source_mask, labels=label_ids, return_dict=True)
        loss = out_dict[0]
        logits = out_dict[1]
        return loss, logits

    def consistency_loss(self, logits_batches: torch.Tensor, 
                            consistency_sentences_input_ids_batches: torch.Tensor, 
                            labels_batches: torch.Tensor) -> torch.Tensor:
        loss_func = nn.CrossEntropyLoss()
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
        if self.use_consistency_loss:
            loss = self.consistency_loss(
                logits, 
                batch["consistency_sentences_input_ids"].to(device=self.device, dtype=torch.long),
                batch["target_input_ids"].to(device=self.device, dtype=torch.long)
            )
        elif self.use_sf_loss:
            sf_loss = custom_loss.word_based_semantic_fidelity_loss_with_confidences(
                batch["source_data_values"],
                batch["target_text"],
                logits,
                self.tokenizer
            )
            loss = (1 - self.sf_loss_alpha) * loss + self.sf_loss_alpha * loss * sf_loss
        elif self.use_dcs_loss:
            dcs = custom_loss.calculate_dcs(
                logits,
                batch["original_data"],
                self.tokenizer,
                self.current_dataset
            )
            loss = loss + self.dcs_beta * dcs
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



class CustomOPTModel(nn.Module):
    def __init__(self, model, tokenizer, device="cuda", 
        use_consistency_loss: bool = False, consistency_loss_weight: float = 0.1, 
        use_sf_loss: bool = False, sf_loss_alpha: float = 0.5) -> None:
        super(CustomOPTModel, self).__init__()
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.use_consistency_loss = use_consistency_loss
        self.use_sf_loss = use_sf_loss
        if self.use_consistency_loss:
            log.info(f"\n\tUsing consistency loss")
            self.consistency_loss_weight = consistency_loss_weight
        elif self.use_sf_loss:
            log.info(f"\n\tUsing semantic fidelity loss")
            self.sf_loss_alpha = sf_loss_alpha
            log.info(f"\n\tLoss: (1-{self.sf_loss_alpha})*CE + {self.sf_loss_alpha}*SM")

    def _inner_forward(self, batch: Dict[str, torch.Tensor]):
        """
        Args:
            device (str)
            batch (Dict[str, torch.Tensor])

        Returns:
            float: loss value
        """
        source_ids = batch["source_input_ids"].to(self.device, dtype=torch.long)
        source_mask = batch["source_attention_mask"].to(self.device, dtype=torch.long)
        target_ids = batch["target_input_ids"].to(self.device, dtype=torch.long)
        target_ids[target_ids[:, :] == self.tokenizer.pad_token_id] = -100
        log.info(f"input: {batch['source_input_ids'].shape}\n{batch['source_input_ids'][0]}")
        log.info(f"target after -100: {target_ids.shape}\n{target_ids[0]}")
        label_ids = target_ids.to(self.device)
        out_dict = self.model(source_ids, attention_mask=source_mask, labels=label_ids, return_dict=True)
        loss = out_dict[0]
        logits = out_dict[1]
        return loss, logits

    def forward(self, batch: Dict[str, torch.Tensor]):
        loss, logits = self._inner_forward(batch)
        import torch.nn.functional as F
        softmaxes = F.softmax(logits, dim=1)
        _, predictions = torch.max(softmaxes, -1)
        batch_sentences = self.tokenizer.batch_decode(predictions, skip_special_tokens=False)
        log.info(f"gen: {predictions.shape}\n{predictions[0]}")
        log.info(f"gen decoded:\n{batch_sentences[0]}")
        if self.use_consistency_loss:
            loss = self.consistency_loss(
                logits, 
                batch["consistency_sentences_input_ids"].to(device=self.device, dtype=torch.long),
                batch["target_input_ids"].to(device=self.device, dtype=torch.long)
            )
        elif self.use_sf_loss:
            sf_loss = custom_loss.word_based_semantic_fidelity_loss(
                batch["source_data_values"],
                batch["target_text"],
                logits,
                self.tokenizer
            )
            loss = (1 - self.sf_loss_alpha) * loss + self.sf_loss_alpha * sf_loss
            # loss = (1 - self.sf_loss_alpha) * loss + self.sf_loss_alpha * loss * sf_loss
        return loss

    def inference(self, batch: Dict[str, torch.Tensor]):
        source_ids = batch["source_input_ids"].to(self.device, dtype=torch.long)
        # source_mask = batch["source_attention_mask"].to(self.device, dtype=torch.long) #! WARNING REMOVED MASK
        return self.model.generate(
            source_ids, 
            # attention_mask=source_mask, 
            max_length=200,
            num_beams=5,
            early_stopping=True,
            num_return_sequences=5,
        ) #! hardcoded length TODO
