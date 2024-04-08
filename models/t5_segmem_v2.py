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
from models.t5_segmem import T5Config, T5SegMem
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


class T5SegMemV2(T5SegMem):
    """
    V2 appends segmem on encoder_outputs and influence via cross attention
    instead of V1 which directly prepends segmem on decoder_inputs_embeds
    """
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
        
        assert decoder_inputs_embeds is None
        decoder_inputs_embeds = self.decoder_embed_tokens(decoder_input_ids)                # (b, l, d)

        # NOTE: T5SegMemV2 is an initial version that failed.
        # the idea here is, since the batch of decoder_input is sequential,
        # we can use the {b - 1}-th label as memory to inform the decoding of the {b}-th element.
        # Eventually we find that this would not work, because following `SlakhDataset`'s sampling method,
        # the {b - 1}-th and {b}-th element is not contiguous (see how `length` affects `_split_frame` function in `__getitem__`)
        # so, we need a contiguous version to make this work, which results in the T5SegMemV2WithPrev model 
        
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

        hidden_states = torch.cat([
            hidden_states,
            segmem_embeds_agg, 
        ], dim=1)

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
        return outs_lst