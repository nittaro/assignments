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
        self.input_pj_q = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.input_pj_k = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.input_pj_v = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.output_pj = nn.Linear(embed_dim, embed_dim, bias=bias)

    def forward(self, Q, K, V):
        attns = []
        qs = torch.chunk(self.input_pj_q(Q), self.num_heads, dim=-1)
        ks = torch.chunk(self.input_pj_k(K), self.num_heads, dim=-1)
        vs = torch.chunk(self.input_pj_v(V), self.num_heads, dim=-1)
        for (q, k, v) in zip(qs, ks, vs):
            attns.append(SDA(q, k, v, dropout_p=self.dropout))
        attns = torch.cat(attns, dim=-1)
        
        return self.output_pj(attns)

def SDA(Q, K, V, mask=None, dropout_p=None):
    d = Q.size(-1)
    scale = 1 / sqrt(d)
    attn_weight = torch.matmul(Q, K.transpose(-2, -1)) * scale
    if mask is not None:
        if mask.dtype == torch.bool:
            attn_weight.masked_fill_(mask.logical_not(), float("-inf"))
        else:
            attn_weight += mask
    attn_weight = F.softmax(attn_weight, dim=-1)
    if (dropout_p is not None) and isinstance(dropout_p, float):
        dropout = nn.Dropout(p=dropout_p, inplace=False)
        attn_weight = dropout(attn_weight)
    return torch.matmul(attn_weight, V)

if __name__ == "__main__":
    """
    Q = torch.randn(10, 4, 64)
    K = torch.randn(10, 4, 64)
    V = torch.randn(10, 4, 64)
    sda1 = F.scaled_dot_product_attention(Q, K, V)
    sda2 = SDA(Q, K, V)
    print(sda1[0])
    print(sda2[0])
    """

    embed_dim = 64
    num_heads = 4
    MHA1 = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
    MHA2 = NittaMHA(embed_dim, num_heads)
    in_weight = torch.randn(3 * embed_dim, embed_dim)
    in_bias = torch.randn(3 * embed_dim)
    out_weight = torch.randn(embed_dim, embed_dim)
    out_bias = torch.randn(embed_dim)
    MHA1.in_proj_weight = nn.Parameter(in_weight.clone().detach().requires_grad_(True))
    MHA1.in_proj_bias = nn.Parameter(in_bias.clone().detach().requires_grad_(True))
    MHA1.out_proj.weight = nn.Parameter(out_weight.clone().detach().requires_grad_(True))
    MHA1.out_proj.bias = nn.Parameter(out_bias.clone().detach().requires_grad_(True))

    weight1, weight2, weight3 = torch.chunk(in_weight, 3, dim=0)
    bias1, bias2, bias3 = torch.chunk(in_bias, 3, dim=0)
    MHA2.input_pj_q.weight = nn.Parameter(weight1.clone().detach().requires_grad_(True))
    MHA2.input_pj_q.bias = nn.Parameter(bias1.clone().detach().requires_grad_(True))
    MHA2.input_pj_k.weight = nn.Parameter(weight2.clone().detach().requires_grad_(True))
    MHA2.input_pj_k.bias = nn.Parameter(bias2.clone().detach().requires_grad_(True))
    MHA2.input_pj_v.weight = nn.Parameter(weight3.clone().detach().requires_grad_(True))
    MHA2.input_pj_v.bias = nn.Parameter(bias3.clone().detach().requires_grad_(True))
    MHA2.output_pj.weight = nn.Parameter(out_weight.clone().detach().requires_grad_(True))
    MHA2.output_pj.bias = nn.Parameter(out_bias.clone().detach().requires_grad_(True))

    batch_size = 9
    seq_len = 10
    X = torch.randn(batch_size, seq_len, embed_dim)
    Q = K = V = X
    attn1 = MHA1(Q, K, V, average_attn_weights=False)[0]
    attn2 = MHA2(Q, K, V)
    print(attn1[0, 0])
    print(attn2[0, 0])
