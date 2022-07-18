import logging
from typing import Dict, Optional, Tuple, Union

import torch
import torch.nn as nn
from datatuner.lm.custom import custom_loss
from torch import nn
from torch.nn import CrossEntropyLoss
from transformers import T5ForConditionalGeneration
from transformers.modeling_outputs import BaseModelOutput, Seq2SeqLMOutput

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__file__)


class DatatunerModel(nn.Module):
    def __init__(self, model, tokenizer, device="cuda", current_dataset: str ="",
        use_consistency_loss: bool = False, consistency_loss_weight: float = 0.1, 
        use_sf_loss: bool = False, sf_loss_alpha: float = 0.5,
        use_dcs_loss: bool = False, dcs_beta: float = 1.0) -> None:
        super(DatatunerModel, self).__init__()
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
            log.info(f"\n\tLoss: (1-{self.sf_loss_alpha})*CE + {self.sf_loss_alpha}*SFC")
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
        out_dict = self.model(source_ids, attention_mask=source_mask, labels=label_ids, return_dict=True)
        # out_dict = self.model.custom_forward(source_ids, attention_mask=source_mask, labels=label_ids, return_dict=True)
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
        # softmaxes = nn.functional.softmax(logits, dim=1)
        # _, predictions = torch.max(softmaxes, -1)
        # batch_sentences = self.tokenizer.batch_decode(predictions, skip_special_tokens=False)
        # log.info(f"gen:\n{batch['source_data_values'][0]}")
        # log.info(f"gen:\n{batch_sentences[0]}")
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
            loss = (1 - self.sf_loss_alpha) * loss + self.sf_loss_alpha * sf_loss
        elif self.use_dcs_loss:
            dcs = custom_loss.calculate_dcs(
                logits,
                batch["original_data"],
                self.tokenizer,
                self.current_dataset
            )
            loss = loss + self.dcs_beta * dcs
        return loss, logits
    
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


class GenForwardT5Model(T5ForConditionalGeneration):
    def custom_forward(
            self,
            input_ids: Optional[torch.LongTensor] = None,
            attention_mask: Optional[torch.FloatTensor] = None,
            decoder_input_ids: Optional[torch.LongTensor] = None,
            decoder_attention_mask: Optional[torch.BoolTensor] = None,
            head_mask: Optional[torch.FloatTensor] = None,
            decoder_head_mask: Optional[torch.FloatTensor] = None,
            cross_attn_head_mask: Optional[torch.Tensor] = None,
            encoder_outputs: Optional[Tuple[Tuple[torch.Tensor]]] = None,
            past_key_values: Optional[Tuple[Tuple[torch.Tensor]]] = None,
            inputs_embeds: Optional[torch.FloatTensor] = None,
            decoder_inputs_embeds: Optional[torch.FloatTensor] = None,
            labels: Optional[torch.LongTensor] = None,
            use_cache: Optional[bool] = None,
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            return_dict: Optional[bool] = None,
        ) -> Union[Tuple[torch.FloatTensor], Seq2SeqLMOutput]:
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        #! do this once in case we use normal backprop at the end of all sentences, otherwise redo this after each word?
        # Encode if needed (training, first prediction pass)
        if encoder_outputs is None:
            # Convert encoder inputs in embeddings if needed
            encoder_outputs = self.encoder(
                input_ids=input_ids,
                attention_mask=attention_mask,
                inputs_embeds=inputs_embeds,
                head_mask=head_mask,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )
        elif return_dict and not isinstance(encoder_outputs, BaseModelOutput):
            encoder_outputs = BaseModelOutput(
                last_hidden_state=encoder_outputs[0],
                hidden_states=encoder_outputs[1] if len(encoder_outputs) > 1 else None,
                attentions=encoder_outputs[2] if len(encoder_outputs) > 2 else None,
            )

        hidden_states = encoder_outputs[0]

        if self.model_parallel:
            torch.cuda.set_device(self.decoder.first_device)

        if labels is not None and decoder_input_ids is None and decoder_inputs_embeds is None:
            # get decoder inputs from shifting lm labels to the right
            decoder_input_ids = self._shift_right(labels)

        # Set device for model parallelism
        if self.model_parallel:
            torch.cuda.set_device(self.decoder.first_device)
            hidden_states = hidden_states.to(self.decoder.first_device)
            if decoder_input_ids is not None:
                decoder_input_ids = decoder_input_ids.to(self.decoder.first_device)
            if attention_mask is not None:
                attention_mask = attention_mask.to(self.decoder.first_device)
            if decoder_attention_mask is not None:
                decoder_attention_mask = decoder_attention_mask.to(self.decoder.first_device)

        sequence_len = decoder_input_ids.size(-1) # first should be batch, second sequence (sentence) len
        curr_decoder_attention_mask = None
        curr_decoder_inputs_embeds = None
        lm_logits = tuple()
        #* start of generation loop
        for gen_iteration in range(sequence_len):
            #* do not pass the whole decoder_input_ids, just one for each step 
            curr_decoder_input_ids = decoder_input_ids[:, :gen_iteration+1]
            if decoder_attention_mask is not None:
                curr_decoder_attention_mask = decoder_attention_mask[:, :gen_iteration+1, :]
            if decoder_inputs_embeds is not None:
                curr_decoder_inputs_embeds = decoder_inputs_embeds[:, :gen_iteration+1, :]
            # log.info(f"curr dec ids size: {curr_decoder_input_ids.size()}")

            decoder_outputs = self.decoder(
                input_ids=curr_decoder_input_ids,
                attention_mask=curr_decoder_attention_mask,
                inputs_embeds=curr_decoder_inputs_embeds, # TODO check if they change between each word (should be true) or if they are always the same and can be passed again
                past_key_values=past_key_values,
                encoder_hidden_states=hidden_states,
                encoder_attention_mask=attention_mask,
                head_mask=decoder_head_mask,
                cross_attn_head_mask=cross_attn_head_mask,
                use_cache=use_cache,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )
            # log.info(f"dec out size: {decoder_outputs[0].size()}")

            #* get the output for the last words of each sentence
            next_token_scores = decoder_outputs[0][:, -1, :]
            del decoder_outputs
            # log.info(f"next_token_scores: {next_token_scores.size()}")

            # Set device for model parallelism
            if self.model_parallel:
                torch.cuda.set_device(self.encoder.first_device)
                self.lm_head = self.lm_head.to(self.encoder.first_device)
                next_token_scores = next_token_scores.to(self.lm_head.weight.device)

            if self.config.tie_word_embeddings:
                # Rescale output before projecting on vocab
                # See https://github.com/tensorflow/mesh/blob/fa19d69eafc9a482aff0b59ddd96b025c0cb207d/mesh_tensorflow/transformer/transformer.py#L586
                next_token_scores = next_token_scores * (self.model_dim**-0.5)

            #* update logits
            current_lm_logits = self.lm_head(next_token_scores)
            del next_token_scores
            # log.info(f"curr logits size: {current_lm_logits.size()}")
            lm_logits += (current_lm_logits,)
            del current_lm_logits 
        # log.info(f"Completed generation at iteration {gen_iteration}")
        lm_logits = torch.stack(lm_logits, 1)
        # log.info(f"Logits size: {lm_logits.size()}")

        
        loss = None
        if labels is not None:
            loss_fct = CrossEntropyLoss(ignore_index=-100)
            loss = loss_fct(lm_logits.view(-1, lm_logits.size(-1)), labels.view(-1))

        if not return_dict:
            output = (lm_logits,) + decoder_outputs[1:] + encoder_outputs
            return ((loss,) + output) if loss is not None else output

        return Seq2SeqLMOutput(
            loss=loss,
            logits=lm_logits,
            # past_key_values=decoder_outputs.past_key_values, # deleting it for memory saving, so we don't have them
            # decoder_hidden_states=decoder_outputs.hidden_states,
            # decoder_attentions=decoder_outputs.attentions,
            # cross_attentions=decoder_outputs.cross_attentions,
            encoder_last_hidden_state=encoder_outputs.last_hidden_state,
            encoder_hidden_states=encoder_outputs.hidden_states,
            encoder_attentions=encoder_outputs.attentions,
        )
