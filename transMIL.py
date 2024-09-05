import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from math import sqrt, isqrt

# Nyström Method is used in the calculation of attention weight
class ModifiedMSA(nn.Module):
    def __init__(self, embed_dim, num_heads, num_landmarks=64):
        super().__init__()

        assert embed_dim % num_heads == 0
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scale = sqrt(self.head_dim)
        self.num_landmarks = num_landmarks

        self.proj_qkv = nn.Linear(embed_dim, embed_dim*3, bias=False)
        self.proj_o = nn.Linear(embed_dim, embed_dim, bias=True)

    def forward(self, x):
        # x : (B, N, D)
        batch_size, num_patch, _ = x.size()
        QKV = torch.transpose(self.proj_qkv(x), 0, 2).reshape(3, self.embed_dim, num_patch, batch_size) # (3, D, N, B)
        Q = torch.transpose(QKV[0], 0, 2)
        K = torch.transpose(QKV[1], 0, 2)
        V = torch.transpose(QKV[2], 0, 2)

        Q = torch.reshape(Q, (batch_size, num_patch, self.num_heads, self.head_dim))
        K = torch.reshape(K, (batch_size, num_patch, self.num_heads, self.head_dim))
        V = torch.reshape(V, (batch_size, num_patch, self.num_heads, self.head_dim))

        Q = torch.transpose(Q, 1, 2) # (B, N, h, D/h) -> (B, h, N, D/h)
        K = torch.transpose(K, 1, 2)
        V = torch.transpose(V, 1, 2)

        Q = Q / self.scale

        # landmarks selection
        segs = num_patch // self.num_landmarks
        r = num_patch % self.num_landmarks
        assert segs != 0

        if r == 0:
            Q_tilde = Q.reshape(batch_size, self.num_heads, self.num_landmarks, segs, self.head_dim).mean(-2)
            K_tilde = K.reshape(batch_size, self.num_heads, self.num_landmarks, segs, self.head_dim).mean(-2)
        else:
            num_k = self.num_landmarks - r

            Q_f = Q[:, :, :num_k*segs, :].reshape(batch_size, self.num_heads, num_k, segs, self.head_dim).mean(-2)
            Q_l = Q[:, :, num_k*segs:, :].reshape(batch_size, self.num_heads, r, segs+1, self.head_dim).mean(-2)
            Q_tilde = torch.concat((Q_f, Q_l), dim=-2)

            K_f = K[:, :, :num_k*segs, :].reshape(batch_size, self.num_heads, num_k, segs, self.head_dim).mean(-2)
            K_l = K[:, :, num_k*segs:, :].reshape(batch_size, self.num_heads, r, segs+1, self.head_dim).mean(-2)
            K_tilde = torch.concat((K_f, K_l), dim=-2)

        kernel1 = F.softmax(torch.matmul(Q, K_tilde.transpose(-1, -2)), dim=-1)
        kernel2 = F.softmax(torch.matmul(Q_tilde, K_tilde.transpose(-1, -2)), dim=-1)
        kernel3 = F.softmax(torch.matmul(Q_tilde, K.transpose(-1, -2)), dim=-1)

        attn_weight = torch.matmul(kernel1, torch.linalg.lstsq(kernel2, kernel3).solution)
        out = torch.matmul(attn_weight, V)
        out = torch.transpose(out, 1, 2) # (B, h, N, D/h) -> (B, N, h, D/h)
        out = torch.reshape(out, (batch_size, num_patch, -1))

        out = self.proj_o(out)

        return out

class PPEG(nn.Module):
    def __init__(self, embed_dim):
        super().__init__()
        self.embed_dim = embed_dim
        self.Conv1 = nn.Conv2d(embed_dim, embed_dim, 3, padding=1)
        self.Conv2 = nn.Conv2d(embed_dim, embed_dim, 5, padding=2)
        self.Conv3 = nn.Conv2d(embed_dim, embed_dim, 7, padding=3)

    def forward(self, x):
        # x : (B, N + 1, D)
        N = x.size()[1] - 1
        N_sqrt = int(sqrt(N))
        assert N_sqrt * N_sqrt == N

        H_c = torch.unsqueeze(x[:, 0, :], dim=0) # (B, 1, D)
        H_f = x[:, 1:, :] # (B, N, D)
        H_f = torch.reshape(H_f, (-1, N_sqrt, N_sqrt, self.embed_dim))
        H_f = torch.permute(H_f, (0, 3, 1, 2)) # (B, D, N_sqrt, N_sqrt)
        H_1 = self.Conv1(H_f)
        H_2 = self.Conv2(H_f)
        H_3 = self.Conv3(H_f)
        H_F = H_f + H_1 + H_2 + H_3
        H_F = torch.permute(H_F, (0, 2, 3, 1)) # (B, N_sqrt, N_sqrt, D)
        H_se = torch.reshape(H_F, (-1, N, self.embed_dim)) # (B, N, D)

        return torch.concat((H_c, H_se), dim=1)

class TPTModule(nn.Module):
    def __init__(self,embed_dim, num_heads):
        super().__init__()

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.LN1 = nn.LayerNorm(embed_dim)
        self.LN2 = nn.LayerNorm(embed_dim)
        self.MSA1 = ModifiedMSA(embed_dim, num_heads)
        self.MSA2 = ModifiedMSA(embed_dim, num_heads)
        self.PPEG = PPEG(embed_dim)

    def forward(self, x):
        x = self.MSA1(self.LN1(x)) + x
        x = self.PPEG(x)
        x = self.MSA2(self.LN2(x)) + x

        return x

class TransMIL(nn.Module):
    def __init__(self, num_heads, num_layers, num_classes, embed_dim=1000):
        super().__init__()

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.num_classes = num_classes

        self.resnet = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
        if embed_dim != 1000:
            self.resnet.fc = nn.Linear(2048, embed_dim)

        self.cls_token = nn.Parameter(torch.randn(1, embed_dim))

        self.encoder = nn.Sequential(*[
            TPTModule(embed_dim, num_heads)
            for _ in range(num_layers)
        ])

        self.mlp_head = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, num_classes)
        )

    def forward(self, x):
        # x : (B, N, C, H, W)
        B, n, channels, height, width = x.size()
        N_sqrt = isqrt(n - 1) + 1
        N = N_sqrt * N_sqrt
        M = N - n

        # feature embedding
        with torch.no_grad():
            x = torch.flatten(x, end_dim=1)
            H = self.resnet(x) # (B * N, D)
            H = torch.reshape(H, (B, n, -1))

        H_s = torch.cat((self.cls_token.repeat(repeats=(B, 1, 1)), H, H[:, :M, :]), dim=1)

        H_s = self.encoder(H_s)

        # MLP
        Y = self.mlp_head(H_s[:, 0])

        return Y

if __name__ == "__main__":
    embed_dim = 1000
    num_heads = 10
    num_layers = 3
    num_patches = 256
    num_classes = 10
    num_channels = 3
    height = 224
    width = 224

    """
    # test code for PPEG
    ppeg = PPEG(embed_dim)
    x = torch.rand(1, 26, 1000)
    out = ppeg(x)
    print(out[0])
    """

    """
    # test code for Nystrom approximation
    msa = ModifiedMSA(embed_dim, num_heads)
    x = torch.rand(1, 256, 1000)
    out = msa(x)
    print(out[0])
    """

    # test code for TransMIL
    x = torch.rand(1, num_patches, num_channels, height, width)
    model = TransMIL(num_heads, num_layers, num_classes)
    pred = model(x)
    print(pred)
