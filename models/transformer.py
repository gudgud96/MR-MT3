import torch
from torch import nn, Tensor
from models.t5 import FixedPositionalEmbedding


class VanillaTransformer(nn.Module):
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

    def forward(
        self,
        src: Tensor,
        tgt: Tensor,
    ):
        tgt_in = self._shift_right(tgt)
        tgt_mask, tgt_padding_mask = self.create_tgt_mask(tgt_in)

        src_emb = self.pos_emb(self.proj(src))
        tgt_emb = self.pos_emb(self.decoder_embed_tokens(tgt_in))

        outs = self.transformer(
            src=src_emb,
            tgt=tgt_emb,
            src_mask=None,
            tgt_mask=tgt_mask,
            memory_mask=None,
            src_key_padding_mask=None,
            tgt_key_padding_mask=tgt_padding_mask,
            memory_key_padding_mask=None,
            src_is_causal=False,
            tgt_is_causal=True,
        )
        return self.lm_head(outs)
    
    def _shift_right(self, input_ids):
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
        tgt_padding_mask = tgt == self.config.pad_token_id

        return tgt_mask, tgt_padding_mask