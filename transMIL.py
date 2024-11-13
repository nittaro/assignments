import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from nystrom_attention import NystromAttention
from torchvision import models
from math import sqrt, isqrt
import timm
from timm.data import resolve_data_config
from timm.data.transforms_factory import create_transform
import time


class TransLayer(nn.Module):
    def __init__(self, norm_layer=nn.LayerNorm, dim=512):
        super().__init__()
        self.norm = norm_layer(dim)
        self.attn = NystromAttention(
            dim = dim,
            dim_head = dim//8,
            heads = 8,
            num_landmarks = 64,    # number of landmarks
            pinv_iterations = 6,    # number of moore-penrose iterations for approximating pinverse. 6 was recommended by the paper
            residual = False,         # whether to do an extra residual with the value or not. supposedly faster convergence if turned on
        )

    def forward(self, x):
        x = x + self.attn(self.norm(x))

        return x

class PPEG(nn.Module):
    def __init__(self, dim=512):
        super(PPEG, self).__init__()
        self.proj = nn.Conv2d(dim, dim, 7, 1, 7//2, groups=dim)
        self.proj1 = nn.Conv2d(dim, dim, 5, 1, 5//2, groups=dim)
        self.proj2 = nn.Conv2d(dim, dim, 3, 1, 3//2, groups=dim)

    def forward(self, x, H, W):
        B, _, C = x.shape
        cls_token, feat_token = x[:, 0], x[:, 1:]
        cnn_feat = feat_token.transpose(1, 2).view(B, C, H, W)
        x = self.proj(cnn_feat)+cnn_feat+self.proj1(cnn_feat)+self.proj2(cnn_feat)
        x = x.flatten(2).transpose(1, 2)
        x = torch.cat((cls_token.unsqueeze(1), x), dim=1)
        return x

class TransMIL(nn.Module):
    def __init__(self, n_classes):
        super(TransMIL, self).__init__()
        self.pos_layer = PPEG(dim=512)
        self._fc1 = nn.Sequential(nn.Linear(1024, 512), nn.ReLU())
        self.cls_token = nn.Parameter(torch.randn(1, 1, 512))
        self.n_classes = n_classes
        self.layer1 = TransLayer(dim=512)
        self.layer2 = TransLayer(dim=512)
        self.norm = nn.LayerNorm(512)
        self._fc2 = nn.Linear(512, self.n_classes)


    def forward(self, x):

        h = x.float() #[B, n, 1024]

        h = self._fc1(h) #[B, n, 512]

        #---->pad
        H = h.shape[1]
        _H, _W = int(np.ceil(np.sqrt(H))), int(np.ceil(np.sqrt(H)))
        add_length = _H * _W - H
        h = torch.cat([h, h[:,:add_length,:]],dim = 1) #[B, N, 512]

        #---->cls_token
        B = h.shape[0]
        cls_tokens = self.cls_token.expand(B, -1, -1)
        h = torch.cat((cls_tokens, h), dim=1)

        #---->Translayer x1
        h = self.layer1(h) #[B, N, 512]

        #---->PPEG
        h = self.pos_layer(h, _H, _W) #[B, N, 512]

        #---->Translayer x2
        h = self.layer2(h) #[B, N, 512]

        #---->cls_token
        h = self.norm(h)[:,0]

        #---->predict
        logits = self._fc2(h) #[B, n_classes]
        Y_hat = torch.argmax(logits, dim=1)
        Y_prob = F.softmax(logits, dim = 1)
        results_dict = {'logits': logits, 'Y_prob': Y_prob, 'Y_hat': Y_hat}
        return results_dict

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
            Q_tilde = torch.cat((Q_f, Q_l), dim=-2)

            K_f = K[:, :, :num_k*segs, :].reshape(batch_size, self.num_heads, num_k, segs, self.head_dim).mean(-2)
            K_l = K[:, :, num_k*segs:, :].reshape(batch_size, self.num_heads, r, segs+1, self.head_dim).mean(-2)
            K_tilde = torch.cat((K_f, K_l), dim=-2)

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

class NysAttention(nn.Module):
    def __init__(self, embed_dim, num_heads=8, num_landmarks=64, qkv_bias=False, proj_drop=0., kernel_size=0):
        super().__init__()

        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scale = self.head_dim ** 0.5
        self.num_landmarks = num_landmarks
        self.kernel_size = kernel_size

        self.qkv = nn.Linear(embed_dim, embed_dim * 3, bias=False)
        self.proj = nn.Linear(embed_dim, embed_dim, bias=True)
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

        V = 1 / torch.max(torch.sum(K, dim = -2), dim = -1).values[:, :, None, None] * K.transpose(-1, -2)

        for _ in range(n_iter):
            KV = torch.matmul(K, V)
            V = torch.matmul(0.25 * V, 13 * I - torch.matmul(KV, 15 * I - torch.matmul(KV, 7 * I - KV)))
        return V

    def forward(self, x):
        B, N, C = x.shape
        assert self.num_landmarks < N

        QKV = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        Q, K, V = QKV[0], QKV[1], QKV[2]
        Q /= self.scale

        segs = N // self.num_landmarks
        r = N % self.num_landmarks
        assert segs != 0

        if r == 0:
            K_landmarks = K.reshape(B, self.num_heads, self.num_landmarks, segs, self.head_dim).mean(dim=-2)
            Q_landmarks = Q.reshape(B, self.num_heads, self.num_landmarks, segs, self.head_dim).mean(dim=-2)
        else:
            num_k = self.num_landmarks - r

            K_landmarks_f = K[:, :, :num_k * segs, :].reshape(B, self.num_heads, num_k, segs, self.head_dim).mean(dim=-2)
            K_landmarks_l = K[:, :, num_k * segs:, :].reshape(B, self.num_heads, r, segs + 1, self.head_dim).mean(dim=-2)
            K_landmarks = torch.cat((K_landmarks_f, K_landmarks_l), dim=-2)

            Q_landmarks_f = Q[:, :, :num_k * segs, :].reshape(B, self.num_heads, num_k, segs, self.head_dim).mean(dim=-2)
            Q_landmarks_l = Q[:, :, num_k * segs:, :].reshape(B, self.num_heads, r, segs + 1, self.head_dim).mean(dim=-2)
            Q_landmarks = torch.cat((Q_landmarks_f, Q_landmarks_l), dim=-2)

        kernel1 = torch.nn.functional.softmax(torch.matmul(Q, K_landmarks.transpose(-1, -2)), dim=-1)
        kernel2 = torch.nn.functional.softmax(torch.matmul(Q_landmarks, K_landmarks.transpose(-1, -2)), dim=-1)
        kernel3 = torch.nn.functional.softmax(torch.matmul(Q_landmarks, K.transpose(-1, -2)), dim=-1)

        SV = torch.matmul(torch.matmul(kernel1, self.iterative_inv(kernel2)), torch.matmul(kernel3, V))
        # another way to calculate Moore-Penrose inverse of a matrix
        # S = torch.matmul(kernel_1, torch.linalg.lstsq(kernel_2, kernel_3).solution)
        # SV = torch.matmul(S, V)

        if self.kernel_size > 0:
            SV += self.conv(V)

        SV = SV.transpose(1, 2).reshape(B, N, -1)
        out = self.proj(SV)
        out = self.proj_drop(out)
        out += V.transpose(1, 2).reshape(B, N, -1)

        return out

class NittaPPEG(nn.Module):
    def __init__(self, embed_dim):
        super().__init__()
        self.embed_dim = embed_dim
        self.Conv1 = nn.Conv2d(embed_dim, embed_dim, 3, padding=1, groups=embed_dim)
        self.Conv2 = nn.Conv2d(embed_dim, embed_dim, 5, padding=2, groups=embed_dim)
        self.Conv3 = nn.Conv2d(embed_dim, embed_dim, 7, padding=3, groups=embed_dim)

    def forward(self, x):
        # x : (B, N + 1, D)
        N = x.size(1) - 1
        N_sqrt = int(sqrt(N))
        assert N_sqrt * N_sqrt == N

        H_c = x[:, :1, :] # (B, 1, D)
        H_f = x[:, 1:, :] # (B, N, D)
        H_f = torch.transpose(H_f, 1, 2).reshape(-1, self.embed_dim, N_sqrt, N_sqrt)
        H_1 = self.Conv1(H_f)
        H_2 = self.Conv2(H_f)
        H_3 = self.Conv3(H_f)
        H_F = H_f + H_1 + H_2 + H_3
        H_F = H_F.reshape(-1, self.embed_dim, N)
        H_se = torch.transpose(H_F, 1, 2) # (B, N, D)

        return torch.cat((H_c, H_se), dim=1)

class TPTModule(nn.Module):
    def __init__(self,embed_dim, num_heads):
        super().__init__()

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.LN1 = nn.LayerNorm(embed_dim)
        self.LN2 = nn.LayerNorm(embed_dim)
        """
        self.MSA1 = ModifiedMSA(embed_dim, num_heads)
        self.MSA2 = ModifiedMSA(embed_dim, num_heads)
        """
        self.PPEG = NittaPPEG(embed_dim)
        self.attn1 = NysAttention(embed_dim)
        self.attn2 = NysAttention(embed_dim)


    def forward(self, x):
        """
        x = self.MSA1(self.LN1(x)) + x
        x = self.PPEG(x)
        x = self.MSA2(self.LN2(x)) + x
        return x
        """

        x = self.attn1(self.LN1(x)) + x
        x = self.PPEG(x)
        x = self.attn2(self.LN2(x)) + x
        return x

class NittaTransMIL(nn.Module):
    def __init__(self, num_heads, num_layers, num_classes, embed_dim=512):
        super().__init__()

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.num_classes = num_classes

        self.feature_extractor = timm.create_model('resnet50', pretrained=True, num_classes=0)
        config = resolve_data_config({}, model=self.feature_extractor)
        self.transform = create_transform(**config)

        self.fc = nn.Sequential(nn.Linear(2048, embed_dim), nn.ReLU())
        self.cls_token = nn.Parameter(torch.randn(1, 1, embed_dim))

        self.encoder = nn.Sequential(*[
            TPTModule(embed_dim, num_heads)
            for _ in range(num_layers)
        ])

        self.mlp_head = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, num_classes)
        )

    def forward(self, x):
        # feature embedding
        B, n, _, _, _ = x.size()

        with torch.no_grad():
            x = torch.flatten(x, end_dim=1)
            x = self.transform(x)
            H = self.feature_extractor(x).reshape(B, n, -1)

        # H : (B, n, D)
        H = self.fc(H)
        N_sqrt = isqrt(n - 1) + 1
        N = N_sqrt * N_sqrt
        M = N - n

        # squaring of sequence
        H_s = torch.cat((self.cls_token.repeat(repeats=(B, 1, 1)), H, H[:, :M, :]), dim=1)
        # H_s = torch.cat((self.cls_token.repeat(repeats=(B, 1, 1)), x, x[:, :M, :]), dim=1)

        # TPT module processing
        H_s = self.encoder(H_s)

        # MLP
        logits = self.mlp_head(H_s[:, 0])
        Y_hat = torch.argmax(logits, dim=1)
        Y_prob = F.softmax(logits, dim=1)
        ret = {'logits': logits, 'Y_prob': Y_prob, 'Y_hat': Y_hat}

        return ret


if __name__ == "__main__":
    embed_dim = 512
    num_heads = 8
    num_layers = 1
    num_patches = 256
    num_classes = 10
    num_landmarks = 64

    """
    ppeg1 = PPEG(embed_dim)
    ppeg2 = NittaPPEG(embed_dim)

    weight_conv1 = torch.randn(embed_dim, 1, 3, 3)
    weight_conv2 = torch.randn(embed_dim, 1, 5, 5)
    weight_conv3 = torch.randn(embed_dim, 1, 7, 7)
    bias_conv1 = torch.randn(embed_dim)
    bias_conv2 = torch.randn(embed_dim)
    bias_conv3 = torch.randn(embed_dim)

    ppeg1.proj.weight = nn.Parameter(weight_conv3.clone().detach())
    ppeg1.proj.bias = nn.Parameter(bias_conv3.clone().detach())
    ppeg1.proj1.weight = nn.Parameter(weight_conv2.clone().detach())
    ppeg1.proj1.bias = nn.Parameter(bias_conv2.clone().detach())
    ppeg1.proj2.weight = nn.Parameter(weight_conv1.clone().detach())
    ppeg1.proj2.bias = nn.Parameter(bias_conv1.clone().detach())
    ppeg2.Conv1.weight = nn.Parameter(weight_conv1.clone().detach())
    ppeg2.Conv1.bias = nn.Parameter(bias_conv1.clone().detach())
    ppeg2.Conv2.weight = nn.Parameter(weight_conv2.clone().detach())
    ppeg2.Conv2.bias = nn.Parameter(bias_conv2.clone().detach())
    ppeg2.Conv3.weight = nn.Parameter(weight_conv3.clone().detach())
    ppeg2.Conv3.bias = nn.Parameter(bias_conv3.clone().detach())

    x = torch.rand(1, num_patches+1, embed_dim)
    out1 = ppeg1(x, 16, 16)
    out2 = ppeg2(x)
    print(out1[0])
    print(out2[0])
    """


    """"
    x = torch.randn(1, num_patches, embed_dim*2)
    model1 = NittaTransMIL(num_heads, num_layers, num_classes)
    model2 = TransMIL(n_classes=num_classes)

    cls_token = torch.randn(1, 1, embed_dim)
    weight_fc = torch.randn(embed_dim, embed_dim*2)
    bias_fc = torch.randn(embed_dim)
    weight_LN1 = torch.randn(embed_dim)
    bias_LN1 = torch.randn(embed_dim)
    weight_LN2 = torch.randn(embed_dim)
    bias_LN2 = torch.randn(embed_dim)
    weight_qkv_1 = torch.randn(embed_dim*3, embed_dim)
    weight_proj_1 = torch.randn(embed_dim, embed_dim)
    bias_proj_1 = torch.randn(embed_dim)
    weight_qkv_2 = torch.randn(embed_dim*3, embed_dim)
    weight_proj_2 = torch.randn(embed_dim, embed_dim)
    bias_proj_2 = torch.randn(embed_dim)
    weight_conv1 = torch.randn(embed_dim, 1, 3, 3)
    weight_conv2 = torch.randn(embed_dim, 1, 5, 5)
    weight_conv3 = torch.randn(embed_dim, 1, 7, 7)
    bias_conv1 = torch.randn(embed_dim)
    bias_conv2 = torch.randn(embed_dim)
    bias_conv3 = torch.randn(embed_dim)
    weight_mlp_norm = torch.randn(embed_dim)
    bias_mlp_norm = torch.randn(embed_dim)
    weight_mlp_fc = torch.randn(num_classes, embed_dim)
    bias_mlp_fc = torch.randn(num_classes)

    model1.cls_token = nn.Parameter(cls_token.clone().detach())
    model1.fc[0].weight = nn.Parameter(weight_fc.clone().detach())
    model1.fc[0].bias = nn.Parameter(bias_fc.clone().detach())
    """
    """
    model1.encoder[0].MSA1.proj_qkv.weight = nn.Parameter(weight_qkv_1.clone().detach())
    model1.encoder[0].MSA1.proj_o.weight = nn.Parameter(weight_proj_1.clone().detach())
    model1.encoder[0].MSA1.proj_o.bias = nn.Parameter(bias_proj_1.clone().detach())
    model1.encoder[0].MSA2.proj_qkv.weight = nn.Parameter(weight_qkv_2.clone().detach())
    model1.encoder[0].MSA2.proj_o.weight = nn.Parameter(weight_proj_2.clone().detach())
    model1.encoder[0].MSA2.proj_o.bias = nn.Parameter(bias_proj_2.clone().detach())
    """
    """
    model1.encoder[0].LN1.weight = nn.Parameter(weight_LN1.clone().detach())
    model1.encoder[0].LN1.bias = nn.Parameter(bias_LN1.clone().detach())
    model1.encoder[0].LN2.weight = nn.Parameter(weight_LN2.clone().detach())
    model1.encoder[0].LN2.bias = nn.Parameter(bias_LN2.clone().detach())
    model1.encoder[0].PPEG.Conv1.weight = nn.Parameter(weight_conv1.clone().detach())
    model1.encoder[0].PPEG.Conv2.weight = nn.Parameter(weight_conv2.clone().detach())
    model1.encoder[0].PPEG.Conv3.weight = nn.Parameter(weight_conv3.clone().detach())
    model1.encoder[0].PPEG.Conv1.bias = nn.Parameter(bias_conv1.clone().detach())
    model1.encoder[0].PPEG.Conv2.bias = nn.Parameter(bias_conv2.clone().detach())
    model1.encoder[0].PPEG.Conv3.bias = nn.Parameter(bias_conv3.clone().detach())
    model1.mlp_head[0].weight = nn.Parameter(weight_mlp_norm.clone().detach())
    model1.mlp_head[0].bias = nn.Parameter(bias_mlp_norm.clone().detach())
    model1.mlp_head[1].weight = nn.Parameter(weight_mlp_fc.clone().detach())
    model1.mlp_head[1].bias = nn.Parameter(bias_mlp_fc.clone().detach())

    model1.encoder[0].attn1.to_qkv.weight = nn.Parameter(weight_qkv_1.clone().detach())
    model1.encoder[0].attn1.to_out[0].weight = nn.Parameter(weight_proj_1.clone().detach())
    model1.encoder[0].attn1.to_out[0].bias = nn.Parameter(bias_proj_1.clone().detach())
    model1.encoder[0].attn2.to_qkv.weight = nn.Parameter(weight_qkv_2.clone().detach())
    model1.encoder[0].attn2.to_out[0].weight = nn.Parameter(weight_proj_2.clone().detach())
    model1.encoder[0].attn2.to_out[0].bias = nn.Parameter(bias_proj_2.clone().detach())

    model2.cls_token = nn.Parameter(cls_token.clone().detach())
    model2.pos_layer.proj.weight = nn.Parameter(weight_conv3.clone().detach())
    model2.pos_layer.proj.bias = nn.Parameter(bias_conv3.clone().detach())
    model2.pos_layer.proj1.weight = nn.Parameter(weight_conv2.clone().detach())
    model2.pos_layer.proj1.bias = nn.Parameter(bias_conv2.clone().detach())
    model2.pos_layer.proj2.weight = nn.Parameter(weight_conv1.clone().detach())
    model2.pos_layer.proj2.bias = nn.Parameter(bias_conv1.clone().detach())
    model2._fc1[0].weight = nn.Parameter(weight_fc.clone().detach())
    model2._fc1[0].bias = nn.Parameter(bias_fc.clone().detach())
    model2.layer1.norm.weight = nn.Parameter(weight_LN1.clone().detach())
    model2.layer1.norm.bias = nn.Parameter(bias_LN1.clone().detach())
    model2.layer2.norm.weight = nn.Parameter(weight_LN2.clone().detach())
    model2.layer2.norm.bias = nn.Parameter(bias_LN2.clone().detach())
    model2.layer1.attn.to_qkv.weight = nn.Parameter(weight_qkv_1.clone().detach())
    model2.layer1.attn.to_out[0].weight = nn.Parameter(weight_proj_1.clone().detach())
    model2.layer1.attn.to_out[0].bias = nn.Parameter(bias_proj_1.clone().detach())
    model2.layer2.attn.to_qkv.weight = nn.Parameter(weight_qkv_2.clone().detach())
    model2.layer2.attn.to_out[0].weight = nn.Parameter(weight_proj_2.clone().detach())
    model2.layer2.attn.to_out[0].bias = nn.Parameter(bias_proj_2.clone().detach())
    model2.norm.weight = nn.Parameter(weight_mlp_norm.clone().detach())
    model2.norm.bias = nn.Parameter(bias_mlp_norm.clone().detach())
    model2._fc2.weight = nn.Parameter(weight_mlp_fc.clone().detach())
    model2._fc2.bias = nn.Parameter(bias_mlp_fc.clone().detach())

    out1 = model1(x)
    out2 = model2(x)
    print(out1)
    print(out2['logits'])
    """

    x = torch.rand(1, num_patches, 3, 512, 512)
    model = NittaTransMIL(num_heads, num_layers, num_classes)
    start = time.perf_counter()
    ret = model(x)
    end = time.perf_counter()
    print(ret)
    print('{:.2f}'.format(end-start))
