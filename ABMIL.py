import torch.nn as nn
import torch.nn.functional as F
import torch
import torchvision.transforms as T
import torchvision.transforms.functional as TF

class AttentionBasedMILPooling(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.fc = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x):
        attn_w = F.softmax(self.fc(x), dim=0)
        out = attn_w.T @ x
        return out

class GatedAttentionBasedMILPooling(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
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
        h = self.fc1(x) * self.gate(x)
        h = self.fc2(h)
        attn_w = F.softmax(h, dim=0)
        out = attn_w.T @ x
        return out

if __name__ == "__main__":
    input_dim = 512
    hidden_dim = 128
    # MILPooling = AttentionBasedMILPooling(input_dim, hidden_dim)
    MILPooling = GatedAttentionBasedMILPooling(input_dim, hidden_dim)
    x = torch.randn((16, 512))
    bag_embed = MILPooling(x)
    print(bag_embed.shape)
    print(bag_embed[:, :10])
