import torch
import torch.nn as nn
import torch.nn.functional as F

class NittaSE(nn.Module):
    def __init__(self, C, r, transform, bias=True):
        assert C % r == 0
        super().__init__()
        self.C = C
        self.r = r
        d = C // r
        self.transform = transform
        self.squeeze = nn.AdaptiveAvgPool2d(1)
        self.excitation = nn.Sequential(
                nn.Linear(C, d, bias=bias),
                nn.ReLU(),
                nn.Linear(d, C, bias=bias),
                nn.Sigmoid(),
        )

    def forward(self, x):
        x = self.transform(x)
        z = self.squeeze(x).squeeze()
        s = self.excitation(z)[..., None, None]
        out = s * x 
        return out

if __name__ == "__main__":
    C = 16
    r = 4
    tr = nn.Conv2d(3, C, kernel_size=2)
    model = NittaSE(C, r, tr)
    img = torch.randn(2, 3, 10, 10)
    print(model(img).shape)
    # print(model(img))
