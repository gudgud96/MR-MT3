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
from models.t5_segmem_v2 import T5Config, T5SegMemV2
from transformers.models.t5.modeling_t5 import Seq2SeqLMOutput, BaseModelOutput, BaseModelOutputWithPastAndCrossAttentions, checkpoint, T5LayerNorm, T5Block
from transformers.utils import logging
import torch.nn as nn
import torch.nn.functional as F
import copy
import torch
from einops import rearrange
from tqdm import tqdm


logger = logging.get_logger(__name__)


@dataclass
class Seq2SeqLMOutputNumInsts(Seq2SeqLMOutput):
    loss_inst: Optional[torch.FloatTensor] = None


class T5SegMemV2WithPrev(T5SegMemV2):
    _keys_to_ignore_on_load_missing = [
        r"encoder\.embed_tokens\.weight",
        r"decoder\.embed_tokens\.weight",
        r"lm_head\.weight",
    ]
    _keys_to_ignore_on_load_unexpected = [
        r"decoder\.block\.0\.layer\.1\.EncDecAttention\.relative_attention_bias\.weight",
    ]

    def __init__(
        self, 
        config: T5Config,
        segmem_num_layers: int = 1,
        segmem_length: int = 64,
    ):
        super().__init__(
            config=config,
            segmem_num_layers=segmem_num_layers,
            segmem_length=segmem_length,
        )
    
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
        targets_prev: Optional[torch.LongTensor] = None,
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
        
        assert self.config.pad_token_id == 0
        targets_prev.masked_fill_(targets_prev == -100, self.config.pad_token_id)
        # print('targets_prev', targets_prev, targets_prev.shape)

        segmem_embeds = self.decoder_embed_tokens(targets_prev)                             # (b, l, d)
        segmem_embeds_agg = self.segmem_encoder(segmem_embeds)[0]                           # (b, l, d)
        segmem_embeds_agg = segmem_embeds_agg[:, :self.segmem_length, :]                    # (b, segmem_length, d)

        hidden_states = torch.cat([
            hidden_states,
            segmem_embeds_agg, 
        ], dim=1)
        # print('hidden_states', hidden_states.shape)

        # Decode
        decoder_outputs = self.decoder(
            input_ids=decoder_input_ids,
            attention_mask=decoder_attention_mask,
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
        targets_prev: Optional[torch.LongTensor] = None,
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
            targets_prev=targets_prev,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        return lm_logits

    def generate_2(self, inputs, max_length=1024, output_hidden_states=False, **kwargs):
        batch_size = inputs.shape[0]
        inputs_embeds = self.proj(inputs)
        encoder_outputs = self.encoder(
            inputs_embeds=inputs_embeds,
            return_dict=True
        )
        hidden_states = encoder_outputs[0]
        
        # Decode
        # In this case, we need to decode each batch sequentially
        bs = hidden_states.size(0)
        segmem_ids = None
        outs_lst = []

        for i in range(bs):
            print(i + 1, '/', bs, end='\r')
            decoder_tokens = torch.zeros((1, 1), dtype=torch.long, device=self.device)          # (b, 1)
            cur_enc = hidden_states[i].unsqueeze(0)

            if i == 0:
                # create dummy segmem ids
                segmem_ids = torch.tensor([
                    0 for _ in range(max_length)
                ]).to(self.device)

                # NOTE: whether to add in tie token? 
                # we add in this version, the rationale is that during training
                # there might be prev_segments that are empty, so we don't need to cater to
                # this edge case
                # segmem_ids[0] = 1
                segmem_ids[0] = 1134        # tie token (1131) + 3 special tokens
                segmem_ids[1] = 1
                segmem_ids = segmem_ids.unsqueeze(0)                                            # (b, max_length)
            else:
                assert segmem_ids is not None
            
            segmem_embeds = self.decoder_embed_tokens(segmem_ids)
            segmem_embeds_agg = self.segmem_encoder(segmem_embeds)[0]                           # (b, max_length, d)

            segmem_embeds_agg = segmem_embeds_agg[:, :self.segmem_length, :]   

            cur_hidden_states = torch.cat([
                hidden_states[i].unsqueeze(0),
                segmem_embeds_agg, 
            ], dim=1)                                                                    # (b, segmem_length + 1, d)
            
            for l in range(max_length):  
                decoder_outputs = self.decoder(
                    input_ids=decoder_tokens,
                    encoder_hidden_states=cur_hidden_states,
                    return_dict=True,
                )
                sequence_output = decoder_outputs[0]
                lm_logits = self.lm_head(sequence_output)[:, -1, :]
                cur = torch.argmax(lm_logits, dim=-1)

                decoder_tokens = torch.cat([decoder_tokens, cur.unsqueeze(1)], dim=1)
                if cur.squeeze().item() == self.config.eos_token_id:
                    break
            
            decoder_tokens = F.pad(
                decoder_tokens,
                (0, max_length - decoder_tokens.shape[1]),
                value=0
            )
            outs_lst.append(decoder_tokens)

            segmem_ids = decoder_tokens
        
        outs_lst = torch.cat(outs_lst, dim=0)
        # print('outs_lst')
        # for elem in outs_lst[0]:
        #     print(elem.item(), end=",")
        # print()
        return outs_lst

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
                use_cache=False,
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
        
        # print('decoder_input_ids_start')
        # for elem in decoder_input_ids_start[0]:
        #     print(elem.item(), end=",")
        # print()
        if output_hidden_states:
            return decoder_input_ids_start, hidden_states
        else:
            return decoder_input_ids_start