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
from models.t5 import T5Config, T5ForConditionalGeneration, T5Stack
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


class T5SegMem(T5ForConditionalGeneration):
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
        super().__init__(config=config)

        # add segmem components
        self.segmem_proj = nn.Linear(self.model_dim, self.model_dim, bias=False)
        segmem_config = copy.deepcopy(config)
        segmem_config.is_decoder = False
        segmem_config.use_cache = False
        segmem_config.is_encoder_decoder = False
        segmem_config.num_layers = segmem_num_layers
        self.segmem_encoder = T5Stack(segmem_config, self.segmem_proj)
        self.segmem_length = segmem_length

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
        # print('inputs_embeds segmem', inputs_embeds[0][0][:20])
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
        # print('hidden_states segmem', hidden_states[0][0][:20])

        if labels is not None and decoder_input_ids is None and decoder_inputs_embeds is None:
            # get decoder inputs from shifting lm labels to the right
            decoder_input_ids = self._shift_right(labels)
        
        assert decoder_inputs_embeds is None
        decoder_inputs_embeds = self.decoder_embed_tokens(decoder_input_ids)                # (b, l, d)

        dummy_tensor = torch.tensor([0 for _ in range(labels.shape[1])]).to(self.device)
        dummy_tensor[0] = 1
        segmem_ids = torch.cat([
            decoder_input_ids[:, 1:], 
            torch.zeros(decoder_input_ids.shape[0], 1).to(self.device)
        ], dim=1)

        segmem_ids = torch.cat([dummy_tensor.unsqueeze(0), segmem_ids[:-1]], dim=0).long()

        segmem_embeds = self.decoder_embed_tokens(segmem_ids)                               # (b, l, d)
        segmem_embeds_agg = self.segmem_encoder(segmem_embeds)[0]                           # (b, l, d)
        segmem_embeds_agg = segmem_embeds_agg[:, :self.segmem_length, :]                    # (b, segmem_length, d)

        decoder_inputs_embeds = torch.cat([
            segmem_embeds_agg,
            decoder_inputs_embeds, 
        ], dim=1)

        # Decode
        decoder_outputs = self.decoder(
            input_ids=None,
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

        sequence_output = decoder_outputs[0]                                                # (b, l + segmem_length, d)
        sequence_output = sequence_output[:, self.segmem_length:, :]                        # (b, l, d)           

        if self.config.tie_word_embeddings:
            # Rescale output before projecting on vocab
            # See https://github.com/tensorflow/mesh/blob/fa19d69eafc9a482aff0b59ddd96b025c0cb207d/mesh_tensorflow/transformer/transformer.py#L586
            sequence_output = sequence_output * (self.model_dim**-0.5)
        
        lm_logits = self.lm_head(sequence_output)
        # print('lm_logits segmem', lm_logits[0][0][:20])
        
        return lm_logits, encoder_outputs, decoder_outputs

    def generate(self, inputs, max_length=1024, output_hidden_states=False, **kwargs):
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
                segmem_ids[0] = 1
                segmem_ids = segmem_ids.unsqueeze(0)                                            # (b, max_length)
            else:
                assert segmem_ids is not None
            
            segmem_embeds = self.decoder_embed_tokens(segmem_ids)
            segmem_embeds_agg = self.segmem_encoder(segmem_embeds)[0]                           # (b, max_length, d)

            segmem_embeds_agg = segmem_embeds_agg[:, :self.segmem_length, :]   

            decoder_embeds = self.decoder_embed_tokens(decoder_tokens)                          # (b, 1, d)
            decoder_embeds = torch.cat([
                segmem_embeds_agg,
                decoder_embeds, 
            ], dim=1)                                                                           # (b, segmem_length + 1, d)

            assert decoder_embeds.shape[1] == self.segmem_length + 1
            
            for l in range(max_length):  
                decoder_outputs = self.decoder(
                    input_ids=None,
                    inputs_embeds=decoder_embeds,
                    encoder_hidden_states=cur_enc,
                    return_dict=True,
                )
                sequence_output = decoder_outputs[0]                                            # (b, l + segmem_length, d)
                sequence_output = sequence_output[:, self.segmem_length:, :]                    # (b, l, d)     

                lm_logits = self.lm_head(sequence_output)[:, -1, :]
                cur = torch.argmax(lm_logits, dim=-1)

                decoder_tokens = torch.cat([decoder_tokens, cur.unsqueeze(1)], dim=1)
                if cur.squeeze().item() == self.config.eos_token_id:
                    break
                
                decoder_embeds = self.decoder_embed_tokens(decoder_tokens)
                decoder_embeds = torch.cat([
                    segmem_embeds_agg,
                    decoder_embeds, 
                ], dim=1)
            
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

    def generate_2(self, inputs, max_length=1024, output_hidden_states=False, **kwargs):
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
            print(l + 1, '/', max_length, end='\r')
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
        
        print('decoder_input_ids_start')
        for elem in decoder_input_ids_start[0]:
            print(elem.item(), end=",")
        print()
        if output_hidden_states:
            return decoder_input_ids_start, hidden_states
        else:
            return decoder_input_ids_start