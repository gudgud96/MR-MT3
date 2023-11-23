# testing only
import torch
from torch import nn
from transformers import T5Config
from einops import rearrange


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


def generate_square_subsequent_mask(sz):
    mask = (torch.triu(torch.ones((sz, sz), device="cuda:0")) == 1).transpose(0, 1)
    mask = (
        mask.float()
        .masked_fill(mask == 0, float("-inf"))
        .masked_fill(mask == 1, float(0.0))
    )
    return mask


class T5ForConditionalGeneration(nn.Module):
    def __init__(self):
        super().__init__()
        self.proj = nn.Linear(512, 512, bias=False)
        self.token_emb = nn.Embedding(1136, 512)
        self.pos_emb = FixedPositionalEmbedding(512)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=512, 
            nhead=8,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=6)
        
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=512, 
            nhead=8,
            batch_first=True,
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=6)
        self.lm_head = nn.Linear(512, 1136, bias=False)
    
    def forward(self, inputs, labels):
        inputs_embeds = self.proj(inputs)
        # tmp = self.pos_emb(
        #     seq=inputs_embeds.shape[1]
        # )
        # inputs_embeds = inputs_embeds + tmp
        encoder_hidden_states = self.encoder(inputs_embeds)

        decoder_input_ids = labels[:, :-1]
        start_ids = torch.zeros(decoder_input_ids.shape[0], 1, device="cuda:0").long()
        decoder_input_ids = torch.cat([
            start_ids, 
            decoder_input_ids
        ],
            dim=1
        )
        decoder_embeds = self.token_emb(decoder_input_ids)
        # dec_pos_emb = self.pos_emb(seq=decoder_embeds.shape[1])
        # decoder_embeds += dec_pos_emb

        decoder_hidden_states = self.decoder(
            tgt=decoder_embeds, 
            memory=encoder_hidden_states,
            tgt_mask=generate_square_subsequent_mask(decoder_embeds.size(1)).type(torch.bool).to("cuda:0"),
        )
        lm_logits = self.lm_head(decoder_hidden_states)
        return lm_logits
    
    def generate(self, inputs, max_length=32):
        batch_size = inputs.shape[0]
        inputs_embeds = self.proj(inputs)
        # tmp = self.pos_emb(
        #     seq=inputs_embeds.shape[1]
        # )
        # inputs_embeds = inputs_embeds + tmp

        encoder_hidden_states = self.encoder(inputs_embeds)

        decoder_input_ids_start = torch.ones((batch_size, 1), dtype=torch.long, device="cuda:0") * \
                                    0
        
        # keep track of which sequences are already finished
        
        for l in range(max_length):
            decoder_embeds = self.token_emb(decoder_input_ids_start)  
            # tmp = self.pos_emb(
            #     seq=decoder_embeds.shape[1]
            # )
            # decoder_embeds = decoder_embeds + tmp

            decoder_outputs = self.decoder(
                tgt=decoder_embeds, 
                memory=encoder_hidden_states,
                tgt_mask=generate_square_subsequent_mask(decoder_embeds.size(1)).type(torch.bool).to("cuda:0"),
            )
            print('decoder_outputs', decoder_outputs.shape)
            lm_logits = self.lm_head(decoder_outputs)
            next_tokens = torch.argmax(lm_logits[:, -1, :].unsqueeze(1), dim=-1)
            print('next_tokens', next_tokens)
            decoder_input_ids_start = torch.cat([decoder_input_ids_start, next_tokens], dim=-1)
        
        print(decoder_input_ids_start)
        return decoder_input_ids_start


model = T5ForConditionalGeneration()
model.cuda()
dummy_input = torch.rand(8, 256, 512).to("cuda:0")
dummy_target = torch.arange(1, 257).long().reshape(8, 32).to("cuda:0").long()
opt = torch.optim.Adam(model.parameters(), lr=5e-4)
loss_fn = nn.CrossEntropyLoss()
for i in range(1000):
    opt.zero_grad()
    output = model(dummy_input, dummy_target)
    loss = loss_fn(output.view(-1, 1136), dummy_target.view(-1))
    loss.backward()
    opt.step()
    print(i, loss.item())
    if i % 10 == 0:
        model.eval()
        with torch.no_grad():
            output = model.generate(dummy_input)
        model.train()