import torch
import torch.nn as nn
import torch.nn.functional as F
from math import sqrt

class NittaMHA(nn.Module):
    def __init__(self, embed_dim, num_heads, dropout=0.0, bias=True):
        assert embed_dim % num_heads == 0
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.dropout = dropout
        d_k = embed_dim // num_heads
        self.d_k = d_k
        self.input_pj_layers_q = []
        self.input_pj_layers_k = []
        self.input_pj_layers_v = []
        for i in range(num_heads):
            self.input_pj_layers_q.append(nn.Linear(embed_dim, d_k, bias=bias))
            self.input_pj_layers_k.append(nn.Linear(embed_dim, d_k, bias=bias))
            self.input_pj_layers_v.append(nn.Linear(embed_dim, d_k, bias=bias))
        self.output_pj_layer = nn.Linear(d_k, embed_dim, bias=bias)

    def forward(self, Q, K, V):
        attns = []
        for i in range(self.num_heads):
            q_m = self.input_pj_layers_q[i](Q)
            k_m = self.input_pj_layers_k[i](K)
            v_m = self.input_pj_layers_v[i](V)
            attns.append(SDA(q_m, k_m, v_m, dropout_p=self.dropout))
        attns = torch.cat(attns, dim=-1)
        return self.output_pj_layer(attns)

def SDA(Q, K, V, mask=None, dropout_p=None):
    d = Q.size(-1)
    scale = 1 / sqrt(d)
    attn_weight = Q @ K.transpose(-1, -2) * scale
    if mask is not None:
        if mask.dtype == torch.bool:
            attn_weight.masked_fill_(mask, 0)
        else:
            attn_weight += mask
    attn_weight = F.softmax(attn_weight, dim=-1)
    if (dropout_p is not None) and isinstance(dropout_p, float):
        dropout = nn.Dropout(p=dropout_p, inplace=False)
        attn_weight = dropout(attn_weight)
    return attn_weight @ V

if __name__ == "__main__":
    embed_dim = 16
    num_heads = 4
    MHA1 = nn.MultiheadAttention(embed_dim, num_heads)
    MHA2 = NittaMHA(embed_dim, num_heads)
    N = 3 * embed_dim * embed_dim
    in_weight = torch.arange(0, N).reshape(-1, embed_dim)
    in_bias = torch.arange(0, N)
    out_weight = torch.arange(0, embed_dim * embed_dim).reshape(embed_dim, -1)
    out_bias = torch.arange(0, embed_dim)
    MHA1.in_proj_weight = in_weight
    MHA1.in_proj_bias = in_bias
    MHA1.out_proj.weight = out_weight
    MHA1.out_proj.bias = out_bias
