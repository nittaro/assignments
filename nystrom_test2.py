import torch
import torch.nn as nn
import torch.nn.functional as F
from nystrom_attention import NystromAttention

class NysAttention(nn.Module):
    def __init__(self, dim, num_heads=8, num_landmarks=64):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** 0.5
        self.landmarks = num_landmarks

    def iterative_inv(self, mat, n_iter = 6):
        I = torch.eye(mat.size(-1), device = mat.device)
        K = mat

        # This is the exact coefficient computation, 1 / ||K||_1, of initialization of Z_0, leading to faster convergence. 
        V = 1 / torch.max(torch.sum(K, dim = -2), dim = -1).values[:, :, None, None] * K.transpose(-1, -2)

        for _ in range(n_iter):
            KV = torch.matmul(K, V)
            V = torch.matmul(0.25 * V, 13 * I - torch.matmul(KV, 15 * I - torch.matmul(KV, 7 * I - KV)))
        return V

    def forward(self, Q, K, V):
        B, N, C = Q.size()
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

        kernel_1 = torch.nn.functional.softmax(torch.matmul(Q, keys_landmarks.transpose(-1, -2)), dim = -1)
        kernel_2 = torch.nn.functional.softmax(torch.matmul(queries_landmarks, keys_landmarks.transpose(-1, -2)), dim = -1)
        kernel_3 = torch.nn.functional.softmax(torch.matmul(queries_landmarks, K.transpose(-1, -2)), dim = -1)

        out = torch.matmul(torch.matmul(kernel_1, self.iterative_inv(kernel_2)), torch.matmul(kernel_3, V))

        out = out.transpose(1, 2).reshape(B, N, -1)

        return out

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

    def forward(self, Q, K, V):
        B, N, _ = Q.size()
        Q = Q.reshape(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        K = K.reshape(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        V = V.reshape(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        Q = Q / self.scale

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
        out = torch.matmul(attn_weight, V)

        out = torch.transpose(out, 1, 2) # (B, h, N, D/h) -> (B, N, h, D/h)
        out = torch.reshape(out, (B, N, -1))

        return out

class MSA(nn.Module):
    def __init__(self, embed_dim, num_heads=8):
        super().__init__()

        assert embed_dim % num_heads == 0
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scale = self.head_dim ** 0.5

    def forward(self, Q, K, V):
        B, N, _ = Q.size()
        Q = Q.reshape(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        K = K.reshape(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        V = V.reshape(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        Q = Q / self.scale

        attn_weight = torch.matmul(Q, torch.transpose(K, -1, -2))
        attn_weight = F.softmax(attn_weight, dim=-1)
        out = torch.matmul(attn_weight, V)

        out = torch.transpose(out, 1, 2) # (B, h, N, D/h) -> (B, N, h, D/h)
        out = torch.reshape(out, (B, N, -1))

        return out

if __name__ == "__main__":
    embed_dim = 1024
    num_heads=8
    num_patches = 300
    num_landmarks=64

    model1 = NysAttention(embed_dim, num_heads=num_heads, num_landmarks=num_landmarks)
    model2 = ModifiedMSA(embed_dim, num_heads=num_heads, num_landmarks=num_landmarks)
    model3 = MSA(embed_dim, num_heads=num_heads)

    x = torch.randn(5, num_patches, embed_dim)
    out1 = model1(x, x, x) # (5, num_patches, embed_dim)
    out2 = model2(x, x, x)
    out3 = model3(x, x, x)
    print(out1[0, 100, :8])
    print(out2[0, 100, :8])
    print(out3[0, 100, :8])
