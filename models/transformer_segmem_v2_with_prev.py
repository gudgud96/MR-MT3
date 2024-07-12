import torch
from torch import nn, Tensor
import torch.nn.functional as F
from models.t5 import FixedPositionalEmbedding


class VanillaTransformerSegMemV2WithPrev(nn.Module):
    def __init__(
        self,
        config,
    ):
        super().__init__()
        self.config = config
        self.transformer = nn.Transformer(
            d_model=config.d_model,
            nhead=config.num_heads,
            num_encoder_layers=config.num_encoder_layers,
            num_decoder_layers=config.num_decoder_layers,
            dim_feedforward=config.d_ff,
            dropout=config.dropout_rate,
            norm_first=True,
            batch_first=True,
        )
        self.lm_head = nn.Linear(config.d_model, config.vocab_size)
        
        self.proj = nn.Linear(config.d_model, config.d_model, bias=False)
        self.decoder_embed_tokens = nn.Embedding(config.vocab_size, config.d_model)
        self.pos_emb = FixedPositionalEmbedding(config.d_model)

        # add segmem components
        # self.segmem_proj = nn.Linear(config.d_model, config.d_model, bias=False)
        self.segmem_encoder = nn.TransformerEncoder(
            encoder_layer=nn.TransformerEncoderLayer(
                d_model=config.d_model,
                nhead=config.num_heads,
                dim_feedforward=config.d_ff,
                dropout=0,
                norm_first=True,
                batch_first=True,
            ),
            num_layers=config.num_segmem_layers
        )
        self.segmem_length = config.segmem_length
        print('segmem length: ', self.segmem_length)
    
    def encode(self, src: Tensor):
        src_emb = self.proj(src)
        src_pos_emb = self.pos_emb(src_emb.shape[1])
        src_emb += src_pos_emb
        return self.transformer.encoder(src_emb, None)

    def decode(self, tgt: Tensor, memory: Tensor, tgt_mask: Tensor):
        # we most likely do not need a tgt_mask during decoding
        # because there will not be padding tokens
        tgt_emb = self.decoder_embed_tokens(tgt)
        tgt_pos_emb = self.pos_emb(tgt_emb.shape[1])
        tgt_emb += tgt_pos_emb

        return self.transformer.decoder(
            tgt_emb, memory, tgt_mask
        )

    def forward(
        self,
        src: Tensor,
        tgt: Tensor,
        targets_prev: Tensor,
    ):
        tgt_in = self._shift_right(tgt)
        tgt_mask, tgt_padding_mask = self.create_tgt_mask(tgt_in)

        hidden_states = self.encode(src)

        assert self.config.pad_token_id == 0
        targets_prev.masked_fill_(targets_prev == -100, self.config.pad_token_id)

        # segmem
        segmem_embeds = self.decoder_embed_tokens(targets_prev)                         # (b, l, d)
        segmem_embeds_pos = self.pos_emb(segmem_embeds.shape[1])
        segmem_embeds += segmem_embeds_pos
        segmem_embeds_agg = self.segmem_encoder(segmem_embeds)                          # (b, l, d)
        segmem_embeds_agg = segmem_embeds_agg[:, :self.segmem_length, :]

        hidden_states = torch.cat([
            hidden_states,
            segmem_embeds_agg,
        ], dim=1)

        # decode
        tgt_emb = self.decoder_embed_tokens(tgt_in)
        tgt_pos_emb = self.pos_emb(tgt_emb.shape[1])
        tgt_emb += tgt_pos_emb

        outs = self.transformer.decoder(
            tgt_emb,
            hidden_states,
            tgt_mask,
        )
        return self.lm_head(outs)
    
    def _shift_right(self, input_ids):
        # NOTE: do note in this case our decoder_start_token_id is 2
        # whereas in T5 it is 0, which is the same as pad_token_id
        decoder_start_token_id = self.config.decoder_start_token_id
        pad_token_id = self.config.pad_token_id

        shifted_input_ids = input_ids.new_zeros(input_ids.shape)
        shifted_input_ids[..., 1:] = input_ids[..., :-1].clone()
        shifted_input_ids[..., 0] = decoder_start_token_id

        shifted_input_ids.masked_fill_(shifted_input_ids == -100, pad_token_id)
        return shifted_input_ids
    
    def generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones((sz, sz))) == 1).transpose(0, 1)
        mask = (
            mask.float()
            .masked_fill(mask == 0, float("-inf"))
            .masked_fill(mask == 1, float(0.0))
        )
        return mask

    def create_tgt_mask(self, tgt):
        tgt_seq_len = tgt.shape[1]

        tgt_mask = self.generate_square_subsequent_mask(tgt_seq_len)
        tgt_mask = tgt_mask.to(tgt.device)
        tgt_padding_mask = tgt == self.config.pad_token_id

        return tgt_mask, tgt_padding_mask

    def generate(self, inputs, max_length=1024, output_hidden_states=False, **kwargs):
        batch_size = inputs.shape[0]
        hidden_states = self.encode(inputs)

        # Decode
        # In this case, we need to decode each batch sequentially
        bs = hidden_states.size(0)
        segmem_ids = None
        outs_lst = []

        for i in range(bs):
            print(i + 1, '/', bs, end='\r')
            # start token for T5 and vanilla transformer is different
            # hence always use `decoder_start_token_id`
            decoder_tokens = torch.ones((1, 1), dtype=torch.long, device=inputs.device) * \
                                self.config.decoder_start_token_id
            cur_enc = hidden_states[i].unsqueeze(0)

            if i == 0:
                # create dummy segmem ids
                segmem_ids = torch.tensor([
                    0 for _ in range(max_length)
                ]).to(inputs.device)

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
            segmem_embeds_pos = self.pos_emb(segmem_embeds.shape[1])
            segmem_embeds += segmem_embeds_pos
            segmem_embeds_agg = self.segmem_encoder(segmem_embeds)                          # (b, l, d)
            segmem_embeds_agg = segmem_embeds_agg[:, :self.segmem_length, :]

            cur_hidden_states = torch.cat([
                hidden_states[i].unsqueeze(0),
                segmem_embeds_agg, 
            ], dim=1)
            
            for l in range(max_length):
                # decode
                tgt_emb = self.decoder_embed_tokens(decoder_tokens)
                tgt_pos_emb = self.pos_emb(tgt_emb.shape[1])
                tgt_emb += tgt_pos_emb

                tgt_mask = (self.generate_square_subsequent_mask(decoder_tokens.size(1))
                    .type(torch.bool)).to(decoder_tokens.device)

                sequence_output = self.transformer.decoder(
                    tgt_emb,
                    cur_hidden_states,
                    tgt_mask,
                )
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