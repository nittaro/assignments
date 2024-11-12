import torch
import torch.nn as nn
import torch.nn.functional as F
from nystrom_attention import NystromAttention
import matplotlib.pyplot as plt

class NysAttention(nn.Module):
    def __init__(self, dim, num_heads=8, num_landmarks=64):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** 0.5
        self.landmarks = num_landmarks

        self.qkv = nn.Linear(dim, dim * 3, bias=False)
        self.proj = nn.Linear(dim, dim, bias=False)

    def iterative_inv(self, mat, n_iter = 6):
        I = torch.eye(mat.size(-1), device = mat.device)
        K = mat

        # This is the exact coefficient computation, 1 / ||K||_1, of initialization of Z_0, leading to faster convergence. 
        V = 1 / torch.max(torch.sum(K, dim = -2), dim = -1).values[:, :, None, None] * K.transpose(-1, -2)

        for _ in range(n_iter):
            KV = torch.matmul(K, V)
            V = torch.matmul(0.25 * V, 13 * I - torch.matmul(KV, 15 * I - torch.matmul(KV, 7 * I - KV)))
        return V

    def forward(self, x):
        # (B, num_heads, N, head_dim)
        B, N, C = x.size()
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        q /= self.scale

        keys_head_dim = k.size(-1)
        segs = N // self.landmarks
        if (N % self.landmarks == 0):
            keys_landmarks = k.reshape(B, self.num_heads, self.landmarks, N // self.landmarks, keys_head_dim).mean(dim = -2)
            queries_landmarks = q.reshape(B, self.num_heads, self.landmarks, N // self.landmarks, keys_head_dim).mean(dim = -2)
        else:
            num_k = (segs + 1) * self.landmarks - N
            keys_landmarks_f = k[:, :, :num_k * segs, :].reshape(B, self.num_heads, num_k, segs, keys_head_dim).mean(dim = -2)
            keys_landmarks_l = k[:, :, num_k * segs:, :].reshape(B, self.num_heads, self.landmarks - num_k, segs + 1, keys_head_dim).mean(dim = -2)
            keys_landmarks = torch.cat((keys_landmarks_f, keys_landmarks_l), dim = -2)

            queries_landmarks_f = q[:, :, :num_k * segs, :].reshape(B, self.num_heads, num_k, segs, keys_head_dim).mean(dim = -2)
            queries_landmarks_l = q[:, :, num_k * segs:, :].reshape(B, self.num_heads, self.landmarks - num_k, segs + 1, keys_head_dim).mean(dim = -2)
            queries_landmarks = torch.cat((queries_landmarks_f, queries_landmarks_l), dim = -2)

        kernel1 = torch.nn.functional.softmax(torch.matmul(q, keys_landmarks.transpose(-1, -2)), dim = -1)
        kernel2 = torch.nn.functional.softmax(torch.matmul(queries_landmarks, keys_landmarks.transpose(-1, -2)), dim = -1)
        kernel3 = torch.nn.functional.softmax(torch.matmul(queries_landmarks, k.transpose(-1, -2)), dim = -1)

        SV = torch.matmul(torch.matmul(kernel1, self.iterative_inv(kernel2)), torch.matmul(kernel3, v))
        SV = SV.transpose(1, 2).reshape(B, N, -1)
        out = self.proj(SV)

        return out, SV

    def forwardQKV(self, Q, K, V):
        B, N, _ = Q.size()
        Q = Q.reshape(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        K = K.reshape(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        V = V.reshape(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        Q /= self.scale

        keys_head_dim = K.size(-1)
        segs = N // self.landmarks
        if (N % self.landmarks == 0):
            keys_landmarks = K.reshape(B, self.num_heads, self.landmarks, N // self.landmarks, keys_head_dim).mean(dim = -2)
            queries_landmarks = Q.reshape(B, self.num_heads, self.landmarks, N // self.landmarks, keys_head_dim).mean(dim = -2)
        else:
            num_k = (segs + 1) * self.landmarks - N
            keys_landmarks_f = K[:, :, :num_k * segs, :].reshape(B, self.num_heads, num_k, segs, keys_head_dim).mean(dim = -2)
            keys_landmarks_l = K[:, :, num_k * segs:, :].reshape(B, self.num_heads, self.landmarks - num_k, segs + 1, keys_head_dim).mean(dim = -2)
            keys_landmarks = torch.cat((keys_landmarks_f, keys_landmarks_l), dim = -2)

            queries_landmarks_f = Q[:, :, :num_k * segs, :].reshape(B, self.num_heads, num_k, segs, keys_head_dim).mean(dim = -2)
            queries_landmarks_l = Q[:, :, num_k * segs:, :].reshape(B, self.num_heads, self.landmarks - num_k, segs + 1, keys_head_dim).mean(dim = -2)
            queries_landmarks = torch.cat((queries_landmarks_f, queries_landmarks_l), dim = -2)

        kernel1 = torch.nn.functional.softmax(torch.matmul(Q, keys_landmarks.transpose(-1, -2)), dim = -1)
        kernel2 = torch.nn.functional.softmax(torch.matmul(queries_landmarks, keys_landmarks.transpose(-1, -2)), dim = -1)
        kernel3 = torch.nn.functional.softmax(torch.matmul(queries_landmarks, K.transpose(-1, -2)), dim = -1)

        SV = torch.matmul(torch.matmul(kernel1, self.iterative_inv(kernel2)), torch.matmul(kernel3, V))
        SV = SV.transpose(1, 2).reshape(B, N, -1)

        return SV

class nystromAttention(nn.Module):
    def __init__(self, dim, num_heads, num_landmarks=64):
        super().__init__()

        self.num_head = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** 0.5
        self.num_landmarks = num_landmarks

        self.qkv = nn.Linear(dim, dim * 3, bias=False)
        self.proj = nn.Linear(dim, dim, bias=False)

    def iterative_inv(self, mat, n_iter = 6):
        I = torch.eye(mat.size(-1), device = mat.device)
        K = mat
        
        # This is the exact coefficient computation, 1 / ||K||_1, of initialization of Z_0, leading to faster convergence. 
        V = 1 / torch.max(torch.sum(K, dim = -2), dim = -1).values[:, :, None, None] * K.transpose(-1, -2)
            
        for _ in range(n_iter):
            KV = torch.matmul(K, V)
            V = torch.matmul(0.25 * V, 13 * I - torch.matmul(KV, 15 * I - torch.matmul(KV, 7 * I - KV)))
        return V

    def forward(self, x):
        # (B, num_heads, N, head_dim)
        B, N, C = x.size()
        QKV = self.qkv(x).reshape(B, N, 3, self.num_head, self.head_dim).permute(2, 0, 3, 1, 4)
        Q, K, V = QKV[0], QKV[1], QKV[2]
        Q /= self.scale

        if self.num_landmarks == N:
            attn = torch.nn.functional.softmax(torch.matmul(Q, K.transpose(-1, -2)), dim = -1)
            X = torch.matmul(attn, V)
        else:
            Q_landmarks = Q.reshape(-1, self.num_head, self.num_landmarks, N // self.num_landmarks, self.head_dim).mean(dim = -2)
            K_landmarks = K.reshape(-1, self.num_head, self.num_landmarks, N // self.num_landmarks, self.head_dim).mean(dim = -2)

            kernel1 = torch.nn.functional.softmax(torch.matmul(Q, K_landmarks.transpose(-1, -2)), dim = -1)
            kernel2 = torch.nn.functional.softmax(torch.matmul(Q_landmarks, K_landmarks.transpose(-1, -2)), dim = -1)
            kernel3 = torch.nn.functional.softmax(torch.matmul(Q_landmarks, K.transpose(-1, -2)), dim = -1)
            SV = torch.matmul(torch.matmul(kernel1, self.iterative_inv(kernel2)), torch.matmul(kernel3, V))

        SV = SV.transpose(1, 2).reshape(B, N, -1)
        out = self.proj(SV)
        return out, SV


    def forwardQKV(self, Q, K, V):
        # (B, N, D)
        B, N, _ = Q.size()
        Q = Q.reshape(B, N, self.num_head, self.head_dim).transpose(1, 2)
        K = K.reshape(B, N, self.num_head, self.head_dim).transpose(1, 2)
        V = V.reshape(B, N, self.num_head, self.head_dim).transpose(1, 2)
        Q /= self.scale

        Q_landmarks = Q.reshape(-1, self.num_head, self.num_landmarks, N // self.num_landmarks, self.head_dim).mean(dim = -2)
        K_landmarks = K.reshape(-1, self.num_head, self.num_landmarks, N // self.num_landmarks, self.head_dim).mean(dim = -2)

        kernel1 = torch.nn.functional.softmax(torch.matmul(Q, K_landmarks.transpose(-1, -2)), dim = -1)
        kernel2 = torch.nn.functional.softmax(torch.matmul(Q_landmarks, K_landmarks.transpose(-1, -2)), dim = -1)
        kernel3 = torch.nn.functional.softmax(torch.matmul(Q_landmarks, K.transpose(-1, -2)), dim = -1)
        SV = torch.matmul(torch.matmul(kernel1, self.iterative_inv(kernel2)), torch.matmul(kernel3, V))

        SV = SV.transpose(1, 2).reshape(B, N, -1)
        return SV

# Nyström Method is used in the calculation of attention weight
class ModifiedMSA(nn.Module):
    def __init__(self, embed_dim, num_heads=8, num_landmarks=64):
        super().__init__()

        assert embed_dim % num_heads == 0
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scale = self.head_dim ** 0.5
        self.num_landmarks = num_landmarks

        self.qkv = nn.Linear(embed_dim, embed_dim * 3, bias=False)
        self.proj = nn.Linear(embed_dim, embed_dim, bias=False)

    def forward(self, x):
        # x : (B, N, D)
        batch_size, num_patch, _ = x.size()

        QKV = torch.transpose(self.qkv(x), 0, 2).reshape(3, self.embed_dim, num_patch, batch_size)
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

        # Nyström approximation
        kernel1 = F.softmax(torch.matmul(Q, K_tilde.transpose(-1, -2)), dim=-1)
        kernel2 = F.softmax(torch.matmul(Q_tilde, K_tilde.transpose(-1, -2)), dim=-1)
        kernel3 = F.softmax(torch.matmul(Q_tilde, K.transpose(-1, -2)), dim=-1)

        attn_weight = torch.matmul(kernel1, torch.linalg.lstsq(kernel2, kernel3).solution)
        SV = torch.matmul(attn_weight, V)

        SV = torch.transpose(SV, 1, 2) # (B, h, N, D/h) -> (B, N, h, D/h)
        SV = torch.reshape(SV, (batch_size, num_patch, -1))
        out = self.proj(SV)

        return out, SV

    def forwardQKV(self, Q, K, V):
        # (B, N, D)
        B, N, _ = Q.size()
        Q = Q.reshape(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        K = K.reshape(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        V = V.reshape(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        Q /= self.scale

        # landmarks selection
        segs = N // self.num_landmarks
        r = N % self.num_landmarks
        assert segs != 0

        if r == 0:
            Q_tilde = Q.reshape(B, self.num_heads, self.num_landmarks, segs, self.head_dim).mean(-2)
            K_tilde = K.reshape(B, self.num_heads, self.num_landmarks, segs, self.head_dim).mean(-2)
        else:
            num_k = self.num_landmarks - r

            Q_f = Q[:, :, :num_k*segs, :].reshape(B, self.num_heads, num_k, segs, self.head_dim).mean(-2)
            Q_l = Q[:, :, num_k*segs:, :].reshape(B, self.num_heads, r, segs+1, self.head_dim).mean(-2)
            Q_tilde = torch.concat((Q_f, Q_l), dim=-2)

            K_f = K[:, :, :num_k*segs, :].reshape(B, self.num_heads, num_k, segs, self.head_dim).mean(-2)
            K_l = K[:, :, num_k*segs:, :].reshape(B, self.num_heads, r, segs+1, self.head_dim).mean(-2)
            K_tilde = torch.concat((K_f, K_l), dim=-2)

        # Nyström approximation
        kernel1 = F.softmax(torch.matmul(Q, K_tilde.transpose(-1, -2)), dim=-1)
        kernel2 = F.softmax(torch.matmul(Q_tilde, K_tilde.transpose(-1, -2)), dim=-1)
        kernel3 = F.softmax(torch.matmul(Q_tilde, K.transpose(-1, -2)), dim=-1)

        attn_weight = torch.matmul(kernel1, torch.linalg.lstsq(kernel2, kernel3).solution)
        SV = torch.matmul(attn_weight, V)

        SV = torch.transpose(SV, 1, 2) # (B, h, N, D/h) -> (B, N, h, D/h)
        SV = torch.reshape(SV, (B, N, -1))

        return SV

class MSA(nn.Module):
    def __init__(self, embed_dim, num_heads=8):
        super().__init__()

        assert embed_dim % num_heads == 0
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scale = self.head_dim ** 0.5

        self.qkv = nn.Linear(embed_dim, embed_dim * 3, bias=False)
        self.proj = nn.Linear(embed_dim, embed_dim, bias=False)

    def forward(self, x):
        batch_size, num_patch, _ = x.size()

        QKV = torch.transpose(self.qkv(x), 0, 2).reshape(3, self.embed_dim, num_patch, batch_size)
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
        attn_weight = torch.matmul(Q, torch.transpose(K, -1, -2))
        attn_weight = F.softmax(attn_weight, dim=-1)
        SV = torch.matmul(attn_weight, V)

        SV = torch.transpose(SV, 1, 2) # (B, h, N, D/h) -> (B, N, h, D/h)
        SV = torch.reshape(SV, (batch_size, num_patch, -1))
        out = self.proj(SV)

        return out, SV

    def forwardQKV(self, Q, K, V):
        # (B, N, D)
        B, N, _ = Q.size()
        Q = Q.reshape(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        K = K.reshape(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        V = V.reshape(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        Q /= self.scale

        attn_weight = torch.matmul(Q, torch.transpose(K, -1, -2))
        attn_weight = F.softmax(attn_weight, dim=-1)
        SV = torch.matmul(attn_weight, V)

        SV = torch.transpose(SV, 1, 2) # (B, h, N, D/h) -> (B, N, h, D/h)
        SV = torch.reshape(SV, (B, N, -1))

        return SV


if __name__ == "__main__":
    embed_dim = 1024
    num_heads=8
    num_patches = 256
    num_landmarks=64
    head_dim = embed_dim // num_heads

    model1 = NysAttention(embed_dim, num_heads=num_heads, num_landmarks=num_landmarks)
    model2 = nystromAttention(embed_dim, num_heads, num_landmarks)
    model3 = ModifiedMSA(embed_dim, num_heads=num_heads, num_landmarks=num_landmarks)
    model4 = MSA(embed_dim, num_heads=num_heads)

    qkv_weight = torch.randn(embed_dim*3, embed_dim)
    proj_weight = torch.randn(embed_dim, embed_dim)

    model1.qkv.weight = nn.Parameter(qkv_weight.clone().detach())
    model1.proj.weight = nn.Parameter(proj_weight.clone().detach())
    model2.qkv.weight = nn.Parameter(qkv_weight.clone().detach())
    model2.proj.weight = nn.Parameter(proj_weight.clone().detach())
    model3.qkv.weight = nn.Parameter(qkv_weight.clone().detach())
    model3.proj.weight = nn.Parameter(proj_weight.clone().detach())
    model4.qkv.weight = nn.Parameter(qkv_weight.clone().detach())
    model4.proj.weight = nn.Parameter(proj_weight.clone().detach())

    x = torch.rand(5, num_patches, embed_dim)
    SV1 = model1.forwardQKV(x, x, x)
    SV2 = model2.forwardQKV(x, x, x)
    SV3 = model3.forwardQKV(x, x, x)
    SV4 = model4.forwardQKV(x, x, x)
    print(SV1[0, 0, :8])
    print(SV2[0, 0, :8])
    print(SV3[0, 0, :8])
    print(SV4[0, 0, :8])
    print("-"*70)

    out1, SV1 = model1(x)
    out2, SV2 = model2(x)
    out3, SV3 = model3(x)
    out4, SV4 = model4(x)

    print(SV1[0, 5, :8])
    print(SV2[0, 5, :8])
    print(SV3[0, 5, :8])
    print(SV4[0, 5, :8])
    print("-"*70)
    print(out1[0, 5, :8])
    print(out2[0, 5, :8])
    print(out3[0, 5, :8])
    print(out4[0, 5, :8])
