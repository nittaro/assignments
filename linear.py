import torch
from math import sqrt

class NittaLinear(torch.nn.Module):
    def __init__(self, in_features: int, out_features: int, bias: bool = True):
        # in_features, out_features : int
        # bias : bool
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        k = sqrt(1 / in_features)
        weight = torch.empty(out_features, in_features).uniform_(-k, k)
        self.weight = torch.nn.Parameter(weight)
        if bias:
            bias = torch.empty(out_features).uniform_(-k, k)
            self.bias = torch.nn.Parameter(bias)
        else:
            self.bias = None

    def forward(self, x):
        if self.bias is not None:
            return x @ self.weight.T + self.bias
        else:
            return x @ self.weight.T

if __name__ == "__main__":
    """
    # validate output
    in_dim = 10
    out_dim = 3
    model = NittaLinear(in_dim, out_dim, bias=True)
    x = torch.arange(0, 200, 1)
    x = x.view(-1, 10).to(torch.float32)
    print(model(x))
    """

    # compare output
    in_dim = 10
    out_dim = 3
    x = torch.arange(0, 200, 1)
    x = x.view(-1, 10).to(torch.float32)
    linear1 = NittaLinear(in_dim, out_dim, bias=True)
    linear2 = torch.nn.Linear(in_dim, out_dim, bias=True)
    weight = torch.nn.Parameter(torch.randn(3, 10).requires_grad_(True))
    bias = torch.nn.Parameter(torch.randn(3).requires_grad_(True))
    linear1.weight = weight
    linear2.weight = weight
    linear1.bias = bias
    linear2.bias = bias
    label = torch.randn(20, 3)
    criterion = torch.nn.MSELoss()
    out1 = linear1(x)
    loss1 = criterion(out1, label)
    loss1.backward()
    print(linear1.weight.grad)
    out2 = linear2(x)
    loss2 = criterion(out2, label)
    loss2.backward()
    print(linear2.weight.grad / 2)
