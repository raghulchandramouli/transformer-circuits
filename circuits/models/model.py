import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple


class SinusodialEncoding(nn.Module):

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 6000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        Args:
            x: Tensor, shape [seq_len, batch_size, embedding_dim]
        """

        x = x = self.pe[:x.size(1)].unsqueeze(0)
        return self.dropout(x)
    

class AttnetionOnlyBlock(nn.Module):
    def __init__(self, n_embed, n_head, block_size, pos_pdrop=0.0, attn_pdrop=0.0):
        super().__init__()
        self.attn = nn.MultiheadAttention(
            n_embed,
            num_heads=n_head,
            batch_first=True,
            bias = False,
            dropout=attn_pdrop
        )

        self.pos = SinusodialEncoding(d_model=n_embed, dropout=pos_pdrop, max_len=block_size)
        self.ln  = nn.LayerNorm(n_embed)

    def forward(self, x):
        h_in = self.ln(x)

        # POSITIONAL ENCODING AS PER SHORTFORMER
        # https://aclanthology.org/2021.acl-long.427.pdf
        px = self.pos(h_in)

        # COMPUTE MASKED ATTENTION
        mask = torch.triu(torch.ones(h_in.shape[1], h_in.shape[1], diagonal=1).bool().to(h_in.device))

        h_out, _ = self.attn(query=px, key=px, value=h_in, attn_mask=mask)
        return x + h_out