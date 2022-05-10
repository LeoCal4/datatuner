from transformers import GPT2LMHeadModel, T5ForConditionalGeneration
from transformers.modeling_outputs import BaseModelOutput, Seq2SeqLMOutput

from typing import Optional, Tuple, Union
import torch
from datatuner.lm.cross_entropy import CrossEntropyLoss
from datatuner.lm.data_loader import MASKED_OUTPUT
from torch.nn import MultiheadAttention, Transformer, LayerNorm


def custom_gpt2_with_smoothing(smoothing=0.0):
    class GPT2LMHeadModelCustom(GPT2LMHeadModel):
        def forward(
            self,
            input_ids,
            past=None,
            attention_mask=None,
            token_type_ids=None,
            position_ids=None,
            head_mask=None,
            labels=None,
        ):

            transformer_outputs = self.transformer(
                input_ids, past=past, token_type_ids=token_type_ids, position_ids=position_ids, head_mask=head_mask
            )

            hidden_states = transformer_outputs[0]

            lm_logits = self.lm_head(hidden_states)

            outputs = (lm_logits,) + transformer_outputs[1:]
            if labels is not None:
                # Shift so that tokens < n predict n
                shift_logits = lm_logits[..., :-1, :].contiguous()
                shift_labels = labels[..., 1:].contiguous()
                # Flatten the tokens
                loss_fct = CrossEntropyLoss(ignore_index=MASKED_OUTPUT, smooth_eps=smoothing, reduction="mean")

                loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))

                outputs = (loss,) + outputs

            return outputs  # (loss), lm_logits, presents, (all hidden_states), (attentions)

    return GPT2LMHeadModelCustom


def custom_gpt2_with_agumented_attention(
    num_heads: int = 16, normalize_lm_output: bool = False, normalize_attention_sum: bool = False, device="cuda",
    ):
    class GPT2LMHeadModelCustom(GPT2LMHeadModel):
        def __init__(self, config):
            super().__init__(config)
            self.transpose_input = False
            try:
                self.final_attention = MultiheadAttention(config.n_embd, num_heads, batch_first=True)
            # TypeError: __init__() got an unexpected keyword argument 'batch_first'
            except TypeError:
                # this is added in case the pytorch version is too old and does not support batch_first
                # the code to handle this case is taken from (forward method): 
                # https://pytorch.org/docs/1.10/_modules/torch/nn/modules/activation.html#MultiheadAttention.forward
                self.transpose_input = True
                self.final_attention = MultiheadAttention(config.n_embd, num_heads)
            if normalize_lm_output:
                self.gpt2_output_norm = LayerNorm(config.n_embd)
            if normalize_attention_sum:
                self.attention_sum_norm = LayerNorm(config.n_embd)

        def forward(
            self,
            input_ids=None,
            past=None,
            attention_mask=None,
            token_type_ids=None,
            position_ids=None,
            head_mask=None,
            labels=None,
        ):
            # print(f"input_ids shape: {input_ids.shape}")
            # print(f"token_type_ids shape: {token_type_ids.shape}")
            transformer_outputs = self.transformer(
                    input_ids, past=past, token_type_ids=token_type_ids, position_ids=position_ids, head_mask=head_mask
            )
            hidden_states = transformer_outputs[0]
            # print(f"hidden_states shape: {hidden_states.shape}")

            hidden_states = torch.squeeze(hidden_states, 1)
            # print(f"Transformer hidden states post-squeeze: {hidden_states.shape}")
            
            # TODO optimization: calculate the hidden_state shape in init (if possible) and create the mask there
            try:
                final_attention_mask = Transformer.generate_square_subsequent_mask(hidden_states.shape[-2]).to(device)
            except TypeError: # TypeError: generate_square_subsequent_mask() missing 1 required positional argument: 'sz'
                sz = hidden_states.shape[-2]
                final_attention_mask = torch.triu(torch.full((sz, sz), float('-inf')), diagonal=1).to(device)

            # print("final_attention_mask.shape: ", final_attention_mask.shape)
            # print("hidden_states.shape before transpose: ", hidden_states.shape)
            if self.transpose_input:
                hidden_states = hidden_states.transpose(1, 0)

            if normalize_lm_output:
                hidden_states = self.gpt2_output_norm(hidden_states)
            # print("========================================================")
            # print("final_attention_mask.shape: ", final_attention_mask.shape)
            # print("hidden_states.shape after transpose: ", hidden_states.shape)
            # print("========================================================")
            final_attention_output, _ = self.final_attention(
                hidden_states, hidden_states, hidden_states, attn_mask=final_attention_mask
            )

            if self.transpose_input:
                # restore hidden states to original state
                hidden_states = hidden_states.transpose(1, 0)
                # transpose attention output
                final_attention_output = final_attention_output.transpose(1, 0)

            # print(f"Final attention output shape: {final_attention_output.shape}")
            # print(f"==== SIZES PRE SUM: hidden {hidden_states.shape} - attn {final_attention_output.shape}")
            final_attention_output = torch.add(hidden_states, final_attention_output)
            # print(f"==== SIZE AFTER SUM: {final_attention_output.shape}")

            if normalize_attention_sum:
                # print(f"==== SIZE PRE LAYER NORM: {final_attention_output.shape}")
                final_attention_output = self.attention_sum_norm(final_attention_output)

            final_attention_output = torch.unsqueeze(final_attention_output, 1)
            lm_logits = self.lm_head(final_attention_output)
            outputs = (lm_logits,) + transformer_outputs[1:]
            if labels is not None:
                # Shift so that tokens < n predict n
                shift_logits = lm_logits[..., :-1, :].contiguous()
                shift_labels = labels[..., 1:].contiguous()
                # Flatten the tokens
                loss_fct = CrossEntropyLoss(ignore_index=MASKED_OUTPUT)
                loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
                outputs = (loss,) + outputs
            return outputs
    return GPT2LMHeadModelCustom


# def custom_t5_with_agumented_attention():
    # pass
def custom_t5_with_agumented_attention(
    attention_in_last_layer=False, num_final_heads: int = 16, normalize_lm_output: bool = False, normalize_attention_sum: bool = False, 
    device="cuda",
    ):
    class T5ForConditionalGenerationCustom(T5ForConditionalGeneration):
        def __init__(self, config):
            super().__init__(config)
            self.transpose_input = False
            if attention_in_last_layer:
                try:
                    self.final_attention = MultiheadAttention(config.d_model, num_final_heads, batch_first=True)
                # TypeError: __init__() got an unexpected keyword argument 'batch_first'
                except TypeError as e:
                    print(e)
                    # this is added in case the pytorch version is too old and does not support batch_first
                    # the code to handle this case is taken from (forward method): 
                    # https://pytorch.org/docs/1.10/_modules/torch/nn/modules/activation.html#MultiheadAttention.forward
                    self.transpose_input = True
                    self.final_attention = MultiheadAttention(config.d_model, num_final_heads)
                #* this makes sense only with final attention
                if normalize_attention_sum:
                    self.attention_sum_norm = LayerNorm(config.d_model)
            if normalize_lm_output:
                self.t5_output_norm = LayerNorm(config.d_model)



        def forward(
            self,
            input_ids: Optional[torch.LongTensor] = None,
            attention_mask: Optional[torch.FloatTensor] = None,
            decoder_input_ids: Optional[torch.LongTensor] = None,
            decoder_attention_mask: Optional[torch.BoolTensor] = None,
            head_mask: Optional[torch.FloatTensor] = None,
            decoder_head_mask: Optional[torch.FloatTensor] = None,
            cross_attn_head_mask: Optional[torch.Tensor] = None,
            encoder_outputs: Optional[Tuple[Tuple[torch.Tensor]]] = None,
            past_key_values: Optional[Tuple[Tuple[torch.Tensor]]] = None, # * past (?)
            inputs_embeds: Optional[torch.FloatTensor] = None,
            decoder_inputs_embeds: Optional[torch.FloatTensor] = None,
            labels: Optional[torch.LongTensor] = None,
            use_cache: Optional[bool] = None,
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            return_dict: Optional[bool] = False,
        ):
            """Taken from:
            https://github.com/huggingface/transformers/blob/v4.18.0/src/transformers/models/t5/modeling_t5.py#L1539
            """
            use_cache = use_cache if use_cache is not None else self.config.use_cache
            return_dict = return_dict if return_dict is not None else self.config.use_return_dict

            # FutureWarning: head_mask was separated into two input args - head_mask, decoder_head_mask
            if head_mask is not None and decoder_head_mask is None:
                if self.config.num_layers == self.config.num_decoder_layers:
                    # warnings.warn(__HEAD_MASK_WARNING_MSG, FutureWarning)
                    decoder_head_mask = head_mask

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

            # Decode
            decoder_outputs = self.decoder(
                input_ids=decoder_input_ids,
                attention_mask=decoder_attention_mask,
                inputs_embeds=decoder_inputs_embeds,
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

            sequence_output = decoder_outputs[0]

            # Set device for model parallelism
            if self.model_parallel:
                torch.cuda.set_device(self.encoder.first_device)
                self.lm_head = self.lm_head.to(self.encoder.first_device)
                sequence_output = sequence_output.to(self.lm_head.weight.device)

            if self.config.tie_word_embeddings: # TODO move after attention?
                # Rescale output before projecting on vocab
                # See https://github.com/tensorflow/mesh/blob/fa19d69eafc9a482aff0b59ddd96b025c0cb207d/mesh_tensorflow/transformer/transformer.py#L586
                sequence_output = sequence_output * (self.model_dim**-0.5)

            if attention_in_last_layer:
                try:
                    final_attention_mask = Transformer.generate_square_subsequent_mask(sequence_output.shape[-2]).to(device)
                except TypeError: # TypeError: generate_square_subsequent_mask() missing 1 required positional argument: 'sz'
                    sz = sequence_output.shape[-2]
                    final_attention_mask = torch.triu(torch.full((sz, sz), float('-inf')), diagonal=1).to(device)

            if self.transpose_input:
                # print("transposing seq output")
                sequence_output = sequence_output.transpose(1, 0)
            
            if normalize_lm_output:
                # print("normalizing t5 output")
                sequence_output = self.t5_output_norm(sequence_output)

            if attention_in_last_layer:
                final_attention_output, _ = self.final_attention(
                    sequence_output, sequence_output, sequence_output, attn_mask=final_attention_mask
                )

            if self.transpose_input:
                # restore hidden states to original state
                sequence_output = sequence_output.transpose(1, 0)
                # transpose attention output
                final_attention_output = final_attention_output.transpose(1, 0)
            
            lm_head_output = sequence_output
            if attention_in_last_layer:
                lm_head_output = torch.add(sequence_output, final_attention_output)
            
            if normalize_attention_sum:
                lm_head_output = self.attention_sum_norm(final_attention_output)

            lm_logits = self.lm_head(lm_head_output)

            loss = None
            if labels is not None:
                loss_fct = CrossEntropyLoss(ignore_index=-100) # TODO pass the right one
                loss = loss_fct(lm_logits.view(-1, lm_logits.size(-1)), labels.view(-1))

            if not return_dict:
                output = (lm_logits,) + decoder_outputs[1:] + encoder_outputs
                return ((loss,) + output) if loss is not None else output

            return Seq2SeqLMOutput(
                loss=loss,
                logits=lm_logits,
                past_key_values=decoder_outputs.past_key_values,
                decoder_hidden_states=decoder_outputs.hidden_states,
                decoder_attentions=decoder_outputs.attentions,
                cross_attentions=decoder_outputs.cross_attentions,
                encoder_last_hidden_state=encoder_outputs.last_hidden_state,
                encoder_hidden_states=encoder_outputs.hidden_states,
                encoder_attentions=encoder_outputs.attentions,
            )

    return T5ForConditionalGenerationCustom