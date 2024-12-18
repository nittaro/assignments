import torch
import torch.nn as nn
import torch.nn.functional as F
from nystrom_attention import NystromAttention

class NysAttention(nn.Module):
    def __init__(self, embed_dim, num_heads=8, num_landmarks=64, qkv_bias=False, attn_drop=0., proj_drop=0., kernel_size=0):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scale = self.head_dim ** 0.5
        self.num_landmarks = num_landmarks
        self.kernel_size = kernel_size

        self.qkv = nn.Linear(embed_dim, embed_dim * 3, bias=False)
        self.proj = nn.Linear(embed_dim, embed_dim, bias=True)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj_drop = nn.Dropout(proj_drop)

        if self.kernel_size > 0:
            self.conv = nn.Conv2d(
                in_channels = self.num_heads,
                out_channels = self.num_heads,
                kernel_size = (self.kernel_size, 1),
                padding = (self.kernel_size // 2, 0),
                bias = False,
                groups = self.num_heads,
            )

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
        B, N, C = x.shape
        QKV = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        Q, K, V = QKV[0], QKV[1], QKV[2]
        Q /= self.scale

        segs = N // self.num_landmarks
        if (N % self.num_landmarks == 0):
            K_landmarks = K.reshape(B, self.num_heads, self.num_landmarks, N // self.num_landmarks, self.head_dim).mean(dim=-2)
            Q_landmarks = Q.reshape(B, self.num_heads, self.num_landmarks, N // self.num_landmarks, self.head_dim).mean(dim=-2)
        else:
            num_k = (segs + 1) * self.num_landmarks - N
            K_landmarks_f = K[:, :, :num_k * segs, :].reshape(B, self.num_heads, num_k, segs, self.head_dim).mean(dim=-2)
            K_landmarks_l = K[:, :, num_k * segs:, :].reshape(B, self.num_heads, self.num_landmarks - num_k, segs + 1, self.head_dim).mean(dim=-2)
            K_landmarks = torch.cat((K_landmarks_f, K_landmarks_l), dim=-2)

            Q_landmarks_f = Q[:, :, :num_k * segs, :].reshape(B, self.num_heads, num_k, segs, self.head_dim).mean(dim=-2)
            Q_landmarks_l = Q[:, :, num_k * segs:, :].reshape(B, self.num_heads, self.num_landmarks - num_k, segs + 1, self.head_dim).mean(dim=-2)
            Q_landmarks = torch.cat((Q_landmarks_f, Q_landmarks_l), dim=-2)

        kernel1 = torch.nn.functional.softmax(torch.matmul(Q, K_landmarks.transpose(-1, -2)), dim=-1)
        kernel2 = torch.nn.functional.softmax(torch.matmul(Q_landmarks, K_landmarks.transpose(-1, -2)), dim=-1)
        kernel3 = torch.nn.functional.softmax(torch.matmul(Q_landmarks, K.transpose(-1, -2)), dim=-1)

        SV = torch.matmul(torch.matmul(kernel1, self.iterative_inv(kernel2)), torch.matmul(kernel3, V))

        if self.kernel_size > 0:
            SV += self.conv(V)

        SV = SV.transpose(1, 2).reshape(B, N, -1)
        out = self.proj(SV)
        out = self.proj_drop(out)
        out += V.transpose(1, 2).reshape(B, N, -1)

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

        # Nyström approximation
        kernel1 = F.softmax(torch.matmul(Q, K_tilde.transpose(-1, -2)), dim=-1)
        kernel2 = F.softmax(torch.matmul(Q_tilde, K_tilde.transpose(-1, -2)), dim=-1)
        kernel3 = F.softmax(torch.matmul(Q_tilde, K.transpose(-1, -2)), dim=-1)

        attn_weight = torch.matmul(kernel1, torch.linalg.lstsq(kernel2, kernel3).solution)
        out = torch.matmul(attn_weight, V)

        out = torch.transpose(out, 1, 2) # (B, h, N, D/h) -> (B, N, h, D/h)
        out = torch.reshape(out, (batch_size, num_patch, -1))
        out = self.proj_o(out)

        return out

class MSA(nn.Module):
    def __init__(self, embed_dim, num_heads=8):
        super().__init__()

        assert embed_dim % num_heads == 0
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scale = self.head_dim ** 0.5

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
        attn_weight = torch.matmul(Q, torch.transpose(K, -1, -2))
        attn_weight = F.softmax(attn_weight, dim=-1)
        out = torch.matmul(attn_weight, V)

        out = torch.transpose(out, 1, 2) # (B, h, N, D/h) -> (B, N, h, D/h)
        out = torch.reshape(out, (batch_size, num_patch, -1))
        out = self.proj_o(out)

        return out

if __name__ == "__main__":
    embed_dim = 1024
    num_heads=8
    num_patches = 300
    num_landmarks=64

    model1 = NysAttention(embed_dim, num_heads=num_heads, num_landmarks=num_landmarks)
    model2 = ModifiedMSA(embed_dim, num_heads=num_heads, num_landmarks=num_landmarks)
    model3 = NystromAttention(dim=embed_dim, dim_head=128, heads=num_heads, num_landmarks=num_landmarks, residual=False)
    model4 = MSA(embed_dim, num_heads=num_heads)
    model5 = nn.MultiheadAttention(embed_dim, num_heads, bias=False, batch_first=True)

    weight_qkv = torch.randn(3*embed_dim, embed_dim)
    weight_proj = torch.randn(embed_dim, embed_dim)

    model1.qkv.weight = nn.Parameter(weight_qkv.clone().detach())
    model1.proj.weight = nn.Parameter(weight_proj.clone().detach())
    model2.proj_qkv.weight = nn.Parameter(weight_qkv.clone().detach())
    model2.proj_o.weight = nn.Parameter(weight_proj.clone().detach())
    model3.to_qkv.weight = nn.Parameter(weight_qkv.clone().detach())
    model3.to_out[0].weight = nn.Parameter(weight_proj.clone().detach())
    model3.to_out[0].bias = nn.Parameter(torch.zeros(embed_dim))
    model4.proj_qkv.weight = nn.Parameter(weight_qkv.clone().detach())
    model4.proj_o.weight = nn.Parameter(weight_proj.clone().detach())
    model5.in_proj_weight = nn.Parameter(weight_qkv.clone().detach())
    model5.out_proj.weight = nn.Parameter(weight_proj.clone().detach())

    x = torch.randn(5, num_patches, embed_dim)
    out1 = model1(x)
    out2 = model2(x)
    out3 = model3(x)
    out4 = model4(x)
    out5, _ = model5(x, x, x)
    print(out1[0, 0, :5])
    print(out2[0, 0, :5])
    print(out3[0, 0, :5])
    print(out4[0, 0, :5])
    print(out5[0, 0, :5])
