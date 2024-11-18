import torch.nn as nn
import torch.nn.functional as F
import torch
import torchvision.transforms as T
import torchvision.transforms.functional as TF

class AttentionBasedMILPooling(nn.Module):
    def __init__(self, input_dim, hidden_dim, need_weights=False):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.need_weights = need_weights
        self.fc = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x):
        # (batch_size, num_instances, feature_dim)
        attn_w = torch.transpose(F.softmax(self.fc(x), dim=1), -1, -2) # (batch_size, 1, num_instances)
        out = attn_w @ x # (batch_size, 1, feature_dim)

        if self.need_weights:
            return out.squeeze(1), attn_w
        else:
            return out.squeeze(1)

class GatedAttentionBasedMILPooling(nn.Module):
    def __init__(self, input_dim, hidden_dim, need_weights=False):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.need_weights = need_weights
        self.fc1 = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Tanh(),
        )
        self.gate = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Sigmoid(),
        )

        self.fc2 = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        # (batch_size, num_instances, feature_dim)
        h = self.fc1(x) * self.gate(x)
        h = self.fc2(h)
        attn_w = torch.transpose(F.softmax(h, dim=1), -1, -2) # (batch_size, 1, num_instances)
        out = attn_w @ x # (batch_size, 1, feature_dim)

        if self.need_weights:
            return out.squeeze(1), attn_w
        else:
            return out.squeeze(1)

if __name__ == "__main__":
    input_dim = 512
    hidden_dim = 128
    # MILPooling = AttentionBasedMILPooling(input_dim, hidden_dim)
    MILPooling = GatedAttentionBasedMILPooling(input_dim, hidden_dim)
    x = torch.randn(4, 16, 512)
    bag_embed = MILPooling(x)
    print(bag_embed.shape)
    print(bag_embed[:, :10])
