import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from math import sqrt

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
    Q = torch.rand(3, 4, 5)
    V = K = Q
    out1 = SDA(Q, K, V)
    out2 = F.scaled_dot_product_attention(Q, K, V)
    # print(out1 == out2)
    # print(out1)
    # print(out2)
    print(out1 - out2)
