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
    ):
        tgt_in = self._shift_right(tgt)
        tgt_mask, tgt_padding_mask = self.create_tgt_mask(tgt_in)

        # src is a spectrogram, so we use nn.Linear here
        src_emb = self.proj(src)
        src_pos_emb = self.pos_emb(src_emb.shape[1])
        src_emb += src_pos_emb

        tgt_emb = self.decoder_embed_tokens(tgt_in)
        tgt_pos_emb = self.pos_emb(tgt_emb.shape[1])
        tgt_emb += tgt_pos_emb

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

        decoder_input_ids_start = torch.ones((batch_size, 1), dtype=torch.long, device=inputs.device) * \
                                    self.config.decoder_start_token_id
        
        # keep track of which sequences are already finished
        unfinished_sequences = torch.ones(batch_size, dtype=torch.long, device=inputs.device)
        eos_token_id_tensor = torch.tensor(self.config.eos_token_id).to(inputs.device)
        
        for l in range(max_length):
            # generate causal mask for autoregressive decoding
            tgt_mask = (self.generate_square_subsequent_mask(decoder_input_ids_start.size(1))
                    .type(torch.bool)).to(decoder_input_ids_start.device)
            
            sequence_output = self.decode(
                tgt=decoder_input_ids_start,
                memory=hidden_states,
                tgt_mask=tgt_mask,
            )
            
            lm_logits = self.lm_head(sequence_output)
            next_tokens = torch.argmax(lm_logits[:, -1, :].unsqueeze(1), dim=-1)

            next_tokens = next_tokens * unfinished_sequences.unsqueeze(-1) + self.config.pad_token_id * (1 - unfinished_sequences.unsqueeze(-1))
            eos_indices = torch.where(next_tokens == self.config.eos_token_id)[0]
            unfinished_sequences[eos_indices] = 0
            decoder_input_ids_start = torch.cat([decoder_input_ids_start, next_tokens], dim=-1)

            # stop when each sentence is finished
            if unfinished_sequences.max() == 0:
                break
                    
        return decoder_input_ids_start