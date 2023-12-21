# Copy from https://github.com/huggingface/transformers/blob/main/src/transformers/models/t5/modeling_t5.py
#
# coding=utf-8
# Copyright 2018 Mesh TensorFlow authors, T5 Authors and HuggingFace Inc. team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from typing import Optional, Tuple, Union
from dataclasses import dataclass
from transformers import T5Config, T5PreTrainedModel
from transformers.models.t5.modeling_t5 import Seq2SeqLMOutput, BaseModelOutput, BaseModelOutputWithPastAndCrossAttentions, checkpoint, T5LayerNorm, T5Block
from transformers.utils import logging
import torch.nn as nn
import copy
import torch
from einops import rearrange
from tqdm import tqdm


logger = logging.get_logger(__name__)


@dataclass
class Seq2SeqLMOutputNumInsts(Seq2SeqLMOutput):
    loss_inst: Optional[torch.FloatTensor] = None


class T5ForConditionalGeneration(T5PreTrainedModel):
    _keys_to_ignore_on_load_missing = [
        r"encoder\.embed_tokens\.weight",
        r"decoder\.embed_tokens\.weight",
        r"lm_head\.weight",
    ]
    _keys_to_ignore_on_load_unexpected = [
        r"decoder\.block\.0\.layer\.1\.EncDecAttention\.relative_attention_bias\.weight",
    ]

    def __init__(self, config: T5Config):
        super().__init__(config)
        self.model_dim = config.d_model
        # NOTE: temporary change, for MT3 please uncomment this line
        self.proj = nn.Linear(self.model_dim, self.model_dim, bias=False)
        
        # NOTE: for encodec model please uncomment this line
        # self.proj = nn.Embedding(
        #     config.encoder_vocab_size, config.d_model)

        self.decoder_embed_tokens = nn.Embedding(
            config.vocab_size, config.d_model)

        encoder_config = copy.deepcopy(config)
        encoder_config.is_decoder = False
        encoder_config.use_cache = False
        encoder_config.is_encoder_decoder = False
        self.encoder = T5Stack(encoder_config, self.proj, "encoder")

        decoder_config = copy.deepcopy(config)
        decoder_config.is_decoder = True
        decoder_config.is_encoder_decoder = False
        decoder_config.num_layers = config.num_decoder_layers
        self.decoder = T5Stack(decoder_config, self.decoder_embed_tokens, "decoder")

        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)
        # self.mean_pool = nn.AdaptiveAvgPool1d(1)
        # self.num_inst_cls = nn.Linear(config.d_model, 16)

        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):
        return self.decoder_embed_tokens

    def set_input_embeddings(self, new_embeddings):
        self.decoder_embed_tokens = new_embeddings
        self.encoder.set_input_embeddings(new_embeddings)
        self.decoder.set_input_embeddings(new_embeddings)

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    def get_output_embeddings(self):
        return self.lm_head

    def get_encoder(self):
        return self.encoder

    def get_decoder(self):
        return self.decoder

    def get_model_outputs(
        self,
        inputs: Optional[torch.FloatTensor] = None,
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
    ):
        # FutureWarning: head_mask was separated into two input args - head_mask, decoder_head_mask
        if head_mask is not None and decoder_head_mask is None:
            if self.config.num_layers == self.config.num_decoder_layers:
                decoder_head_mask = head_mask
        if inputs is not None:
            inputs_embeds = self.proj(inputs)
        # print('inputs_embeds', inputs_embeds[0][0][:20])
        # Encode if needed (training, first prediction pass)
        if encoder_outputs is None:
            # Convert encoder inputs in embeddings if needed
            encoder_outputs = self.encoder(
                input_ids=None,
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
                hidden_states=encoder_outputs[1] if len(
                    encoder_outputs) > 1 else None,
                attentions=encoder_outputs[2] if len(
                    encoder_outputs) > 2 else None,
            )

        hidden_states = encoder_outputs[0]

        if labels is not None and decoder_input_ids is None and decoder_inputs_embeds is None:
            # get decoder inputs from shifting lm labels to the right
            decoder_input_ids = self._shift_right(labels)

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

        if self.config.tie_word_embeddings:
            # Rescale output before projecting on vocab
            # See https://github.com/tensorflow/mesh/blob/fa19d69eafc9a482aff0b59ddd96b025c0cb207d/mesh_tensorflow/transformer/transformer.py#L586
            sequence_output = sequence_output * (self.model_dim**-0.5)
        
        lm_logits = self.lm_head(sequence_output)

        # mean_hidden_states = self.mean_pool(sequence_output.transpose(1, 2)).squeeze(-1)
        # inst_cls_logits = self.num_inst_cls(mean_hidden_states)
        
        return lm_logits, encoder_outputs, decoder_outputs

    def forward(
        self,
        inputs: Optional[torch.FloatTensor] = None,
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
        num_insts: Optional[torch.LongTensor] = None,
    ) -> Union[Tuple[torch.FloatTensor], Seq2SeqLMOutput]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for computing the sequence classification/regression loss. Indices should be in `[-100, 0, ...,
            config.vocab_size - 1]`. All labels set to `-100` are ignored (masked), the loss is only computed for
            labels in `[0, ..., config.vocab_size]`
        Returns:
        Examples:
        ```python
        >>> from transformers import T5Tokenizer, T5ForConditionalGeneration
        >>> tokenizer = T5Tokenizer.from_pretrained("t5-small")
        >>> model = T5ForConditionalGeneration.from_pretrained("t5-small")
        >>> # training
        >>> input_ids = tokenizer("The <extra_id_0> walks in <extra_id_1> park", return_tensors="pt").input_ids
        >>> labels = tokenizer("<extra_id_0> cute dog <extra_id_1> the <extra_id_2>", return_tensors="pt").input_ids
        >>> outputs = model(input_ids=input_ids, labels=labels)
        >>> loss = outputs.loss
        >>> logits = outputs.logits
        >>> # inference
        >>> input_ids = tokenizer(
        ...     "summarize: studies have shown that owning a dog is good for you", return_tensors="pt"
        >>> ).input_ids  # Batch size 1
        >>> outputs = model.generate(input_ids)
        >>> print(tokenizer.decode(outputs[0], skip_special_tokens=True))
        >>> # studies have shown that owning a dog is good for you.
        ```"""
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        
        lm_logits, encoder_outputs, decoder_outputs = self.get_model_outputs(
            inputs=inputs,
            attention_mask=attention_mask,
            decoder_input_ids=decoder_input_ids,
            decoder_attention_mask=decoder_attention_mask,
            head_mask=head_mask,
            decoder_head_mask=decoder_head_mask,
            cross_attn_head_mask=cross_attn_head_mask,
            encoder_outputs=encoder_outputs,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            decoder_inputs_embeds=decoder_inputs_embeds,
            labels=labels,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        return lm_logits
    
    def generate(self, inputs, max_length=1024, output_hidden_states=False, **kwargs):
        batch_size = inputs.shape[0]
        inputs_embeds = self.proj(inputs)
        encoder_outputs = self.encoder(
            inputs_embeds=inputs_embeds,
            return_dict=True
        )
        hidden_states = encoder_outputs[0]

        decoder_input_ids_start = torch.ones((batch_size, 1), dtype=torch.long, device=self.device) * \
                                    self.config.decoder_start_token_id
        
        # keep track of which sequences are already finished
        unfinished_sequences = torch.ones(batch_size, dtype=torch.long, device=self.device)
        eos_token_id_tensor = torch.tensor(self.config.eos_token_id).to(self.device)
        
        for l in range(max_length):
            decoder_outputs = self.decoder(
                input_ids=decoder_input_ids_start,
                encoder_hidden_states=hidden_states,
                # output_hidden_states=output_hidden_states,
                return_dict=True,
            )

            # handle output hidden states
            # if output_hidden_states:
            #     hidden_states = decoder_outputs.hidden_states # (num_layers + 1, batch_size, 1, hidden_size)
            #     print('hidden states', hidden_states[0].shape)
            #     hidden_states = [torch.cat(layer, dim=0) for layer in hidden_states]
            #     print('dec hidden states', hidden_states[0].shape)
            #     hidden_states = torch.cat(hidden_states, dim=0)
            #     print('dec hidden states', hidden_states.shape)
            
            sequence_output = decoder_outputs[0]
            lm_logits = self.lm_head(sequence_output)
            next_tokens = torch.argmax(lm_logits[:, -1, :].unsqueeze(1), dim=-1)

            next_tokens = next_tokens * unfinished_sequences.unsqueeze(-1) + self.config.pad_token_id * (1 - unfinished_sequences.unsqueeze(-1))
            eos_indices = torch.where(next_tokens == self.config.eos_token_id)[0]
            unfinished_sequences[eos_indices] = 0
            decoder_input_ids_start = torch.cat([decoder_input_ids_start, next_tokens], dim=-1)

            # stop when each sentence is finished
            if unfinished_sequences.max() == 0:
                break
            
            # print(l, decoder_input_ids_start.shape)
        
        if output_hidden_states:
            return decoder_input_ids_start, hidden_states
        else:
            return decoder_input_ids_start
    
    def prepare_inputs_for_generation(
        self,
        input_ids,
        past=None,
        attention_mask=None,
        head_mask=None,
        decoder_head_mask=None,
        cross_attn_head_mask=None,
        use_cache=None,
        encoder_outputs=None,
        **kwargs
    ):

        # cut decoder_input_ids if past is used
        if past is not None:
            input_ids = input_ids[:, -1:]

        return {
            "decoder_input_ids": input_ids,
            "past_key_values": past,
            "encoder_outputs": encoder_outputs,
            "attention_mask": attention_mask,
            "head_mask": head_mask,
            "decoder_head_mask": decoder_head_mask,
            "cross_attn_head_mask": cross_attn_head_mask,
            "use_cache": use_cache,
        }

    def prepare_decoder_input_ids_from_labels(self, labels: torch.Tensor):
        return self._shift_right(labels)

    def _reorder_cache(self, past, beam_idx):
        # if decoder past is not included in output
        # speedy decoding is disabled and no need to reorder
        if past is None:
            logger.warning(
                "You might want to consider setting `use_cache=True` to speed up decoding")
            return past

        reordered_decoder_past = ()
        for layer_past_states in past:
            # get the correct batch idx from layer past batch dim
            # batch dim of `past` is at 2nd position
            reordered_layer_past_states = ()
            for layer_past_state in layer_past_states:
                # need to set correct `past` for each of the four key / value states
                reordered_layer_past_states = reordered_layer_past_states + (
                    layer_past_state.index_select(
                        0, beam_idx.to(layer_past_state.device)),
                )

            assert reordered_layer_past_states[0].shape == layer_past_states[0].shape
            assert len(reordered_layer_past_states) == len(layer_past_states)

            reordered_decoder_past = reordered_decoder_past + \
                (reordered_layer_past_states,)
        return reordered_decoder_past


# ================= adversarial attacks ============== #
# These two methods only noises the input, and expect the output y to stay the same
# This is an end-to-end approach. We did not include noising for the autoregressive part.
# Hence, we are assuming that this method affects more on the encoder, ensuring the encoder output 
# to be resilient to adversarial noise.


class T5Adversarial(T5ForConditionalGeneration):
    def __init__(self, config: T5Config):
        super().__init__(config)
    
    def fgsm(self, inputs, labels, epsilon=0.1):
        delta = torch.zeros_like(inputs, requires_grad=True)
        lm_logits, _, _ = self.get_model_outputs(inputs=inputs, labels=labels)
        loss = nn.CrossEntropyLoss(ignore_index=-100)(
            lm_logits.view(-1, lm_logits.size(-1)), labels.view(-1)
        )
        loss.backward()
        return epsilon * delta.grad.detach().sign()

    def pgd_linf(self, inputs, labels, epsilon=0.1, alpha=0.01, num_iter=5):
        delta = torch.zeros_like(inputs, requires_grad=True)
        
        for _ in range(num_iter):
            lm_logits, _, _ = self.get_model_outputs(inputs=inputs + delta, labels=labels)
            loss = nn.CrossEntropyLoss(ignore_index=-100)(
                lm_logits.view(-1, lm_logits.size(-1)), labels.view(-1)
            )
            loss.backward()
            delta.data = (delta + alpha*delta.grad.detach().sign()).clamp(-epsilon,epsilon)
            delta.grad.zero_()
        return delta.detach()
    
    # def forward(
    #     self,
    #     inputs: Optional[torch.FloatTensor] = None,
    #     attention_mask: Optional[torch.FloatTensor] = None,
    #     decoder_input_ids: Optional[torch.LongTensor] = None,
    #     decoder_attention_mask: Optional[torch.BoolTensor] = None,
    #     head_mask: Optional[torch.FloatTensor] = None,
    #     decoder_head_mask: Optional[torch.FloatTensor] = None,
    #     cross_attn_head_mask: Optional[torch.Tensor] = None,
    #     encoder_outputs: Optional[Tuple[Tuple[torch.Tensor]]] = None,
    #     past_key_values: Optional[Tuple[Tuple[torch.Tensor]]] = None,
    #     inputs_embeds: Optional[torch.FloatTensor] = None,
    #     decoder_inputs_embeds: Optional[torch.FloatTensor] = None,
    #     labels: Optional[torch.LongTensor] = None,
    #     use_cache: Optional[bool] = None,
    #     output_attentions: Optional[bool] = None,
    #     output_hidden_states: Optional[bool] = None,
    #     return_dict: Optional[bool] = None,
    #     attack = "fgsm",
    #     attack_epsilon = 0.1
    # ) -> Union[Tuple[torch.FloatTensor], Seq2SeqLMOutput]:
    #     """
    #     The adversarial version (on encoder should be):

    #     """
    #     use_cache = use_cache if use_cache is not None else self.config.use_cache
    #     return_dict = return_dict if return_dict is not None else self.config.use_return_dict

    #     # if attack == "fgsm":
    #     #     delta = self.fgsm(
    #     #         inputs, labels, epsilon=attack_epsilon
    #     #     )
    #     # else:
    #     delta = torch.zeros_like(inputs, requires_grad=True)

    #     # print("delta", delta)
        
    #     lm_logits, encoder_outputs, decoder_outputs = self.get_model_outputs(
    #         inputs=inputs + delta,
    #         attention_mask=attention_mask,
    #         decoder_input_ids=decoder_input_ids,
    #         decoder_attention_mask=decoder_attention_mask,
    #         head_mask=head_mask,
    #         decoder_head_mask=decoder_head_mask,
    #         cross_attn_head_mask=cross_attn_head_mask,
    #         encoder_outputs=encoder_outputs,
    #         past_key_values=past_key_values,
    #         inputs_embeds=inputs_embeds,
    #         decoder_inputs_embeds=decoder_inputs_embeds,
    #         labels=labels,
    #         use_cache=use_cache,
    #         output_attentions=output_attentions,
    #         output_hidden_states=output_hidden_states,
    #         return_dict=return_dict,
    #     )

    #     loss = None
    #     if labels is not None:
    #         loss_fct = nn.CrossEntropyLoss(ignore_index=-100)
    #         loss = loss_fct(
    #             lm_logits.view(-1, lm_logits.size(-1)), labels.view(-1)
    #         )

    #     if not return_dict:
    #         output = (lm_logits,) + decoder_outputs[1:] + encoder_outputs
    #         print("return dict")
    #         return ((loss,) + output) if loss is not None else output
        
    #     print("return here", loss)
    #     return Seq2SeqLMOutput(
    #         loss=loss,
    #         logits=lm_logits,
    #         past_key_values=decoder_outputs.past_key_values,
    #         decoder_hidden_states=decoder_outputs.hidden_states,
    #         decoder_attentions=decoder_outputs.attentions,
    #         cross_attentions=decoder_outputs.cross_attentions,
    #         encoder_last_hidden_state=encoder_outputs.last_hidden_state,
    #         encoder_hidden_states=encoder_outputs.hidden_states,
    #         encoder_attentions=encoder_outputs.attentions,
    #     )


class T5Stack(T5PreTrainedModel):
    def __init__(self, config, embed_tokens=None, name=""):
        super().__init__(config)

        self.embed_tokens = embed_tokens
        self.is_decoder = config.is_decoder

        self.pos_emb = FixedPositionalEmbedding(config.d_model)

        self.block = nn.ModuleList(
            [T5Block(config, has_relative_attention_bias=False)
             for i in range(config.num_layers)]
        )
        self.final_layer_norm = T5LayerNorm(
            config.d_model, eps=config.layer_norm_epsilon)
        self.dropout = nn.Dropout(config.dropout_rate)

        # Initialize weights and apply final processing
        self.post_init()
        self.gradient_checkpointing = False

        self.name = name

    def get_input_embeddings(self):
        return self.embed_tokens

    def set_input_embeddings(self, new_embeddings):
        self.embed_tokens = new_embeddings

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        inputs_embeds=None,
        head_mask=None,
        cross_attn_head_mask=None,
        past_key_values=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if input_ids is not None and inputs_embeds is not None:
            err_msg_prefix = "decoder_" if self.is_decoder else ""
            raise ValueError(
                f"You cannot specify both {err_msg_prefix}input_ids and {err_msg_prefix}inputs_embeds at the same time"
            )
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
        elif input_ids is not None:
            input_shape = input_ids.size()[:2]

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        batch_size, seq_length = input_shape[:2]

        # required mask seq length can be calculated via length of past
        mask_seq_length = past_key_values[0][0].shape[2] + \
            seq_length if past_key_values is not None else seq_length
        past_key_values_length = past_key_values[0][0].shape[2] if past_key_values is not None else 0
        if use_cache is True:
            assert self.is_decoder, f"`use_cache` can only be set to `True` if {self} is used as a decoder"

        if attention_mask is None:
            attention_mask = torch.ones(
                batch_size, mask_seq_length).to(inputs_embeds.device)

        if self.is_decoder and encoder_attention_mask is None and encoder_hidden_states is not None:
            encoder_seq_length = encoder_hidden_states.shape[1]
            encoder_attention_mask = torch.ones(
                batch_size, encoder_seq_length, device=inputs_embeds.device, dtype=torch.long
            )

        # initialize past_key_values with `None` if past does not exist
        if past_key_values is None:
            past_key_values = [None] * len(self.block)

        # We can provide a self-attention mask of dimensions [batch_size, from_seq_length, to_seq_length]
        # ourselves in which case we just need to make it broadcastable to all heads.
        extended_attention_mask = self.get_extended_attention_mask(
            attention_mask, input_shape, inputs_embeds.device)

        # If a 2D or 3D attention mask is provided for the cross-attention
        # we need to make broadcastable to [batch_size, num_heads, seq_length, seq_length]
        if self.is_decoder and encoder_hidden_states is not None:
            encoder_batch_size, encoder_sequence_length, _ = encoder_hidden_states.size()
            encoder_hidden_shape = (
                encoder_batch_size, encoder_sequence_length)
            if encoder_attention_mask is None:
                encoder_attention_mask = torch.ones(
                    encoder_hidden_shape, device=inputs_embeds.device)
            encoder_extended_attention_mask = self.invert_attention_mask(
                encoder_attention_mask)
        else:
            encoder_extended_attention_mask = None

        # Prepare head mask if needed
        head_mask = self.get_head_mask(head_mask, self.config.num_layers)
        cross_attn_head_mask = self.get_head_mask(
            cross_attn_head_mask, self.config.num_layers)
        present_key_value_states = () if use_cache else None
        all_hidden_states = () if output_hidden_states else None
        all_attentions = () if output_attentions else None
        all_cross_attentions = () if (output_attentions and self.is_decoder) else None
        position_bias = None
        encoder_decoder_position_bias = None

        # pos_emb
        tmp = self.pos_emb(
                seq=inputs_embeds.shape[1], offset=past_key_values_length)
        inputs_embeds = inputs_embeds + tmp

        # print(self.name, 'before', inputs_embeds[0][0][:5])
        hidden_states = self.dropout(inputs_embeds)
        # print(self.name, 'after', hidden_states[0][0][:5])

        for i, (layer_module, past_key_value) in enumerate(zip(self.block, past_key_values)):
            layer_head_mask = head_mask[i]
            cross_attn_layer_head_mask = cross_attn_head_mask[i]
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            if self.gradient_checkpointing and self.training:
                if use_cache:
                    logger.warning(
                        "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`..."
                    )
                    use_cache = False

                def create_custom_forward(module):
                    def custom_forward(*inputs):
                        return tuple(module(*inputs, use_cache, output_attentions))

                    return custom_forward

                layer_outputs = checkpoint(
                    create_custom_forward(layer_module),
                    hidden_states,
                    extended_attention_mask,
                    position_bias,
                    encoder_hidden_states,
                    encoder_extended_attention_mask,
                    encoder_decoder_position_bias,
                    layer_head_mask,
                    cross_attn_layer_head_mask,
                    None,  # past_key_value is always None with gradient checkpointing
                )
            else:
                layer_outputs = layer_module(
                    hidden_states,
                    attention_mask=extended_attention_mask,
                    position_bias=position_bias,
                    encoder_hidden_states=encoder_hidden_states,
                    encoder_attention_mask=encoder_extended_attention_mask,
                    encoder_decoder_position_bias=encoder_decoder_position_bias,
                    layer_head_mask=layer_head_mask,
                    cross_attn_layer_head_mask=cross_attn_layer_head_mask,
                    past_key_value=past_key_value,
                    use_cache=use_cache,
                    output_attentions=output_attentions,
                )

            # layer_outputs is a tuple with:
            # hidden-states, key-value-states, (self-attention position bias), (self-attention weights), (cross-attention position bias), (cross-attention weights)
            if use_cache is False:
                layer_outputs = layer_outputs[:1] + (None,) + layer_outputs[1:]

            hidden_states, present_key_value_state = layer_outputs[:2]
            # if self.name == "decoder":
            #     print(self.name, 'in loop', i, hidden_states[0][0][:5])

            # We share the position biases between the layers - the first layer store them
            # layer_outputs = hidden-states, key-value-states (self-attention position bias), (self-attention weights),
            # (cross-attention position bias), (cross-attention weights)
            position_bias = layer_outputs[2]
            if self.is_decoder and encoder_hidden_states is not None:
                encoder_decoder_position_bias = layer_outputs[4 if output_attentions else 3]
            # append next layer key value states
            if use_cache:
                present_key_value_states = present_key_value_states + \
                    (present_key_value_state,)

            if output_attentions:
                all_attentions = all_attentions + (layer_outputs[3],)
                if self.is_decoder:
                    all_cross_attentions = all_cross_attentions + \
                        (layer_outputs[5],)

        hidden_states = self.final_layer_norm(hidden_states)
        # torch.manual_seed(365)
        hidden_states = self.dropout(hidden_states)

        # Add last layer
        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        if not return_dict:
            return tuple(
                v
                for v in [
                    hidden_states,
                    present_key_value_states,
                    all_hidden_states,
                    all_attentions,
                    all_cross_attentions,
                ]
                if v is not None
            )
        return BaseModelOutputWithPastAndCrossAttentions(
            last_hidden_state=hidden_states,
            past_key_values=present_key_value_states,
            hidden_states=all_hidden_states,
            attentions=all_attentions,
            cross_attentions=all_cross_attentions,
        )


class FixedPositionalEmbedding(nn.Module):
    def __init__(self, dim, max_length=5000):
        super().__init__()
        inv_freq = 1. / (10000 ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer('inv_freq', inv_freq)
        self.max_length = max_length

    def forward(self, seq, offset=0):
        t = torch.arange(self.max_length, device=self.inv_freq.device).type_as(
            self.inv_freq)
        sinusoid_inp = torch.einsum('i , j -> i j', t, self.inv_freq)
        emb = torch.cat((sinusoid_inp.sin(), sinusoid_inp.cos()), dim=-1)
        y = rearrange(emb, 'n d -> () n d')
        y = y[:, offset:offset + seq, :]
        return y
