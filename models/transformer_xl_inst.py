import torch
from torch import nn
from torch.nn import functional as F


class TokenEmbedding(nn.Module):
    """Token embedding layer
    convert the input tokens to vectors of dimension d_embed

    Attributes:
        n_token (int): size of the dictionary of embeddings
        d_embed (int): the size of each embedding vector
    """

    def __init__(self, n_token, d_embed) -> None:
        super().__init__()
        # embedding scale from Section 3.4 in https://arxiv.org/abs/1706.03762
        self.emb_scale = d_embed ** 0.5
        self.emb_layer = nn.Embedding(n_token, d_embed)
        nn.init.kaiming_normal_(self.emb_layer.weight)

    def forward(self, inp):
        embed = self.emb_layer(inp)
        embed.mul_(self.emb_scale)  # apply embedding scale
        return embed


class PositionalEmbeddingXL(nn.Module):
    def __init__(self, demb):
        super(PositionalEmbeddingXL, self).__init__()

        self.demb = demb

        inv_freq = 1 / (10000 ** (torch.arange(0.0, demb, 2.0, device="cuda") / demb))
        self.register_buffer('inv_freq', inv_freq)

    def forward(self, pos_seq):
        """
        Params: 
            pos_seq: [seq_len]

        Return:
            pos_emb: [bs, seq_len, hidden_dim]
        """
        sinusoid_inp = torch.outer(pos_seq, self.inv_freq)
        pos_emb = torch.cat([sinusoid_inp.sin(), sinusoid_inp.cos()], dim=-1)
        return pos_emb[None, :, :]


class FeedForward(nn.Module):
    def __init__(
        self, d_model: int, d_inner: int, dropout: float = 0, glu: bool = False
    ) -> None:
        """
        Position-wise Feed-Forward Networks

        Args:
            d_model (int): dimmensionality of input
            d_inner (int): dimensionality of the inner-layer
            dropout (float, optional): drop out. Defaults to 0.
            glu (bool, optional): use Gate Linear Unit, Default to True.
        """
        super().__init__()

        project_in = nn.Sequential(nn.Linear(d_model, d_inner), nn.GELU())

        self.ff = nn.Sequential(
            project_in, nn.Dropout(dropout), nn.Linear(d_inner, d_model),
        )

    def forward(self, x):
        return self.ff(x)


class MultiHeadAttention(nn.Module):
    def __init__(
        self, 
        d_model, 
        n_head, 
        dropout, 
        is_rel_pos=False
    ):
        super().__init__()
        self.d_model = d_model
        self.n_head = n_head
        self.d_head = d_model // n_head
        self.scale = self.d_head ** -0.5

        self.Wq = nn.Linear(d_model, d_model, bias=False)
        self.Wk = nn.Linear(d_model, d_model, bias=False)
        self.Wv = nn.Linear(d_model, d_model, bias=False)
        self.dropout = nn.Dropout(dropout)

        self.fc = nn.Linear(d_model, d_model)
        
        # relative position weights
        self.is_rel_pos = is_rel_pos
        if is_rel_pos:
            self.r_net = nn.Linear(d_model, d_model, bias=False)

    def forward(
        self, 
        q_in, 
        k_in, 
        v_in, 
        key_mask=None, 
        attn_mask=None, 
        rel_pos=None
    ):
        """
        q_in: [bs, i, hidden_dim]
        k_in: [bs, j, hidden_dim]
        v_in: [bs, j, hidden_dim]
        key_mask: normally used for padding mask, [bs, j]
        attn_mask: normally used for causal mask, [i, j]
        rel_pos: a dict
        """
        b, i = q_in.size(0), q_in.size(1)
        j = k_in.size(1)

        q = self.Wq(q_in)
        k = self.Wk(k_in)
        v = self.Wv(v_in)

        q = q.view(b, i, self.n_head, self.d_head)
        k = k.view(b, j, self.n_head, self.d_head)
        v = v.view(b, j, self.n_head, self.d_head)

        if self.is_rel_pos and rel_pos != None:
            # Transformer-XL relative positional embedding
            r = self.r_net(rel_pos["r"])
            r = r.view(j, self.n_head, self.d_head)

            # content based attention score: terms (a) and (c)
            qu = q + rel_pos["u"]
            AC = torch.einsum("bihd,bjhd->bhij", qu, k)

            # position based attention score: terms (b) and (d) using
            # Efficient Computation of the Attention with Relative Positional Embedding (Appendix B in https://arxiv.org/pdf/1901.02860.pdf)
            qv = q + rel_pos["v"]
            BD = torch.einsum("bihd,jhd->bhij", qv, r)

            # shift trick at Appendix B in https://arxiv.org/pdf/1901.02860.pdf
            BD = self._rel_shift(BD)

            attn_score = AC + BD
            attn_score.mul_(self.scale)  # scaled dot product attention
        else:
            # scaled dot product attention, shape (b, h, i, j)
            attn_score = torch.einsum("bihd, bjhd -> bhij", q, k) * self.scale

        if key_mask != None:
            assert key_mask.shape == (b, j)
            attn_score.masked_fill_(key_mask.view(b, 1, 1, j), 1e-9)

        if attn_mask != None:
            if attn_mask.dim() == 2:
                assert attn_mask.shape == (i, j)
                attn_mask = attn_mask.view(1, 1, i, j)
            elif attn_mask.dim() == 3:
                assert attn_mask.shape == (b, i, j)
                attn_mask = attn_mask.view(b, 1, i, j)
            attn_score.masked_fill_(attn_mask, float("-inf"))
        
        # attention probability
        attn_prob = F.softmax(attn_score, dim=-1)
        attn_prob = self.dropout(attn_prob)

        # attention output
        attn_vec = torch.einsum("bhij,bjhd->bihd", attn_prob, v)
        attn_vec = attn_vec.contiguous().view(b, i, self.d_model)

        # linear projection
        attn_out = self.fc(attn_vec)

        return attn_out
    
    def _rel_shift(self, x):
        # x shape is [b, h, i, j]
        # Time wise shift along side i and j axis
        # [[1,  2,  3,  4],       [[3,  4,  0, (5)]]
        #  [5,  6,  7,  8],   =>   [6,  7,  8,  0]
        #  [9, 10, 11, 12]]        [9, 10, 11, 12]]
        # Note that the (5) in the first row is carried over from the next row, and this is a trash (value from future)
        # This will be masked by `atten_mask`
        b, h, i, j = x.size()
        # pad zeros at the beginning of j axis
        x_padded = F.pad(x, (1, 0))
        x_padded = x_padded.view(b, h, j + 1, i)
        x = x_padded[:, :, 1:].view_as(x)

        return x


class TransformerXL(nn.Module):
    """
    A variant of Transformer-XL (https://arxiv.org/pdf/1901.02860.pdf) with cross-attention to encoder outputs.
    """
    def __init__(
        self,
        n_token: int,
        context_length: int,
        n_layer: int = 6,
        d_model: int = 512,
        n_head: int = 8,
        emb_dropout: float = 0,
        dropout: float = 0,
        attn_dropout: float = 0,
        ff_dropout: float = 0,
        ff_d_inner: int = 1024,
    ):
        super().__init__()

        self.token_embedding = TokenEmbedding(n_token, d_model)
        self.emb_drop = nn.Dropout(emb_dropout)
        self.pos_drop = nn.Dropout(dropout)
        self.context_length = context_length

        d_head = d_model // n_head

        self.positional_embedding = PositionalEmbeddingXL(d_model)
        self.u = nn.Parameter(torch.Tensor(n_head, d_head))
        self.v = nn.Parameter(torch.Tensor(n_head, d_head))
        # the original work does not use this initialization
        # but I find that if I don't use this, gradients will explode =.=
        nn.init.normal_(self.u, mean=0.0, std=0.02)
        nn.init.normal_(self.v, mean=0.0, std=0.02)
        self.rel_pos = {"u": self.u, "v": self.v}

        self.decoder_layers = nn.ModuleList([])
        for _ in range(n_layer):
            # self-attn: causal, padding-masked
            sa_attn_layer = nn.ModuleList()
            sa_attn_layer.append(
                MultiHeadAttention(
                    d_model=d_model,
                    n_head=n_head,
                    dropout=attn_dropout,
                    is_rel_pos=True,
                )
            )
            sa_attn_layer.append(nn.Dropout(dropout))
            sa_attn_layer.append(nn.LayerNorm(d_model))

            # cross-attn: non-causal, non-padding-masked
            ca_attn_layer = nn.ModuleList()
            ca_attn_layer.append(
                MultiHeadAttention(
                    d_model=d_model,
                    n_head=n_head,
                    dropout=attn_dropout,
                    is_rel_pos=False,
                )
            )
            ca_attn_layer.append(nn.Dropout(dropout))
            ca_attn_layer.append(nn.LayerNorm(d_model))

            ff_layer = nn.ModuleList()
            ff_layer.append(
                FeedForward(
                    d_model=d_model, 
                    d_inner=ff_d_inner, 
                    dropout=ff_dropout, 
                )
            )
            ff_layer.append(nn.Dropout(dropout))
            ff_layer.append(nn.LayerNorm(d_model))

            self.decoder_layers.append(
                nn.ModuleList([sa_attn_layer, ca_attn_layer, ff_layer])
            )

        self.out_layer = nn.Linear(d_model, n_token, bias=False)

    def forward(
        self, 
        encoder_outputs,
        decoder_input_ids, 
        mems=None, 
        padding_token=None, 
    ):
        """
        encoder_outputs: (bs, enc_seq_len, d_model)
        decoder_input_ids: (bs, dec_seq_len)
        mems: List[(bs, mem_len, d_model)]
        """
        qlen = decoder_input_ids.size(1)
        mem_len = mems[0].size(1) if mems else 0

        if padding_token != None:
            # print("got padding token finally", padding_token)
            padding_mask = (decoder_input_ids == padding_token)
            if mems == None:
                self.padding_mask = padding_mask
            else:
                self.padding_mask = torch.cat([self.padding_mask, padding_mask], 1)
        else:
            self.padding_mask = None
        
        # generate an attention mask
        attn_mask = self._create_attn_mask(qlen, mem_len)
        attn_mask = attn_mask.cuda()

        # get token embeddings
        decoder_input_embeds = self.emb_drop(self.token_embedding(decoder_input_ids))

        # generate a positional sequence
        pos_seq = torch.arange(
            mem_len + qlen - 1,
            -1,
            -1.0,
            device="cuda"
        )
        # positional embedding matrix
        r = self.positional_embedding(pos_seq)
        r = self.pos_drop(r)
        self.rel_pos["r"] = r

        new_mems = []
        past_mems = []
        past_hidden = None
        x_cur = decoder_input_embeds
        for i, layer in enumerate(self.decoder_layers):
            sa_attn_layer, ca_attn_layer, ff_layer = layer
            if mems:
                # print("mems exists!")
                past_hidden = mems.pop(0)
                mem = torch.cat([past_hidden, x_cur], 1)

            else:
                mem = x_cur
            
            # print(i, 'x_cur', x_cur.shape, 'mem', mem.shape)
            tmp_x_cur = x_cur.clone()
            
            # === self-attention ===
            sa_attn, drop, norm = sa_attn_layer
            out = sa_attn(
                q_in=x_cur,
                k_in=mem,
                v_in=mem,
                key_mask=self.padding_mask,
                attn_mask=attn_mask,
                rel_pos=self.rel_pos,
            )
            x_cur = norm(x_cur + drop(out))
            # print(i, "sa", x_cur.shape)

            # NOTE: in this implementation, for new_mems, we only need the 
            # hidden states for tokens in the current segment
            new_mems.append(tmp_x_cur.detach())
            # print('new_mems', new_mems[0].shape)
            if past_hidden is not None:
                past_mems.append(past_hidden[:, -self.context_length:].detach())
                # print('past_mems', past_mems[0].shape)

            # === cross-attention ===
            ca_attn, drop, norm = ca_attn_layer
            out = ca_attn(
                q_in=x_cur,
                k_in=encoder_outputs,
                v_in=encoder_outputs,
                key_mask=None,
                attn_mask=None,
                rel_pos=None,
            )
            x_cur = norm(x_cur + drop(out))
            # print(i, "ca", x_cur.shape)

            # === feed-forward ===
            ff, drop, norm = ff_layer
            x_cur = norm(x_cur + drop(ff(x_cur)))
            # print(i, "ff", x_cur.shape)
    
        if self.padding_mask is not None:
            self.padding_mask = self.padding_mask[:, -self.context_length :]
        
        x_cur = self.out_layer(x_cur)

        # output layer
        return x_cur, new_mems, past_mems

    def _create_attn_mask(self, qlen, mem_len):
        return torch.triu(
            torch.full((qlen, qlen + mem_len), True), 1 + mem_len
        )


if __name__ == "__main__":
    model = TransformerXL(n_token=128, context_length=128).cuda()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
    
    # x = torch.randint(1, 128, (1, 1024))
    xlst = []
    xlst.append(torch.randint(1, 128, (1, 67)))
    xlst.append(torch.randint(1, 128, (1, 139)))
    xlst.append(torch.randint(1, 128, (1, 111)))
    xlst.append(torch.randint(1, 128, (1, 187)))
    xlst.append(torch.randint(1, 128, (1, 89)))
    xlst.append(torch.randint(1, 128, (1, 32)))
    xlst.append(torch.randint(1, 128, (1, 193)))
    xlst.append(torch.randint(1, 128, (1, 120)))

    for i in range(len(xlst)):
        xlst[i] = torch.cat([torch.zeros(1,1), xlst[i], torch.zeros(1,1)], 1)

    seglen = 128
    enc_out = torch.randn(8, 256, 512)
    
    loss_fct = nn.CrossEntropyLoss()

    # train via segment
    for ep in range(150):
        mem = None
        bs = enc_out.shape[0]
        loss_sum = 0
        for i in range(bs):
            cur_enc = enc_out[i].cuda().unsqueeze(0)
            cur_dec = xlst[i].cuda().long()

            # if ep == 0:
            #     print(i, cur_enc, cur_dec)
            y, mem = model(cur_enc, cur_dec[:, :-1], mem)
            loss = loss_fct(y.view(-1, y.size(-1)), cur_dec[:, 1:].view(-1))
            
            loss_sum += loss / bs
        
        loss_sum.backward()
        optimizer.step()
        optimizer.zero_grad()
        
        avg_loss = loss_sum
        print(ep, "loss", avg_loss)
        
    
    # inference via autoregressive
    model.eval()
    with torch.no_grad():
        mem = None
        outs_lst = []
        
        for i in range(bs):
            decoder_inputs = torch.zeros(1, 1).cuda().long()
            
            cur_enc = enc_out[i].cuda().unsqueeze(0)
            outs = [decoder_inputs]
            cur = decoder_inputs
            
            for j in range(200):
                y, mem = model(cur_enc, cur, mem)
                y = torch.argmax(y, dim=-1)
                outs.append(y)
                cur = y

                if y.squeeze().item() == 0:
                    break
    
            outs = torch.cat(outs, dim=1)
            outs_lst.append(outs.T)

    for i in range(len(outs_lst)):
        print(i)
        print(len(outs_lst[i].squeeze().long()), outs_lst[i].squeeze().long())
        print(len(xlst[i].squeeze().long()), xlst[i].squeeze().long())
        print("====")
    # print([k.shape for k in outs_lst])
    # outs_lst = nn.utils.rnn.pad_sequence(outs_lst, batch_first=True, padding_value=0).squeeze()
    # print('outs_lst', outs_lst.shape)
    # # outs_lst = torch.cat(outs_lst, dim=0)
    
    # from sklearn.metrics import accuracy_score
    # y_pred = outs_lst.long().cpu().detach().numpy()
    # y_true = x.long().cpu().detach().numpy()

    # for i in range(len(y_pred)):
    #     print(i, (y_true[i] == y_pred[i]).sum() / len(y_true[i]))





    
    
