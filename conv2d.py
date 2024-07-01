import torch
import torch.nn as nn
import torch.nn.functional as F
from math import sqrt

class NittaConv2d(nn.Module):
    def __init__(self, C_in, C_out, kernel_size, stride=1, padding=0, bias=True, padding_mode='zeros'):
        super().__init__()
        self.C_in = C_in
        self.C_out = C_out
        self.stride = stride
        self.padding_mode = padding_mode
        
        if isinstance(kernel_size, int):
            self.H_filter = kernel_size
            self.W_filter = kernel_size
        else:
            self.H_filter = kernel_size[0]
            self.W_filter = kernel_size[1]

        if isinstance(padding, int):
            self.pad = (padding, padding, padding, padding)
        else:
            self.pad = (padding[1], padding[1], padding[0], padding[0])

        k = sqrt(1 / (C_in * self.H_filter * self.W_filter))
        weight = torch.empty(C_out, C_in, self.H_filter, self.W_filter).uniform_(-k, k)
        self.weight = nn.Parameter(weight)

        if bias:
            self.bias = nn.Parameter(torch.empty(C_out).uniform_(-k, k))
        else:
            self.bias = None

    def forward(self, img: torch.Tensor):
        col, N, H_out, W_out = img2col(img, self.H_filter, self.W_filter, self.stride, self.pad, self.padding_mode)
        col_W = self.weight.permute(1, 2, 3, 0).reshape(-1, self.C_out)
        # ret = col @ col_W + self.bias
        ret = F.linear(col, col_W.T, bias = self.bias)
        return ret.reshape(N, H_out, W_out, self.C_out).permute(0, 3, 1, 2)

def img2col(img, H_filter, W_filter, stride, pad, padding_mode):
    N, C, H, W = img.shape
    if isinstance(stride, int):
        stride_x, stride_y = stride, stride
    else:
        stride_y = stride[0]
        stride_x = stride[1]

    H_out = (H + 2 * pad[3] - H_filter) // stride_y + 1
    W_out = (W + 2 * pad[0] - W_filter) // stride_x + 1
    
    if padding_mode == 'zeros':
        img = F.pad(img, pad, "constant", 0)
    else:
        img = F.pad(img, pad, padding_mode)

    col = torch.empty(N, C, H_out, W_out, H_filter, W_filter)

    # expansion
    for y in range(H_filter):
        y_max = y + H_out * stride_y
        for x in range(W_filter):
            x_max = x + W_out * stride_x
            col[:, :, :, :, y, x] = img[:, :, y:y_max:stride_y, x:x_max:stride_x]

    col = col.permute(0, 2, 3, 1, 4, 5).reshape(N * H_out * W_out, -1)
    return col, N, H_out, W_out

if __name__ == "__main__":
    """
    # test code for img2col
    img = torch.arange(0, 27).reshape(1, 3, 3, 3).to(torch.float32)
    out1 = img2col(img, 2, 2, 1, (1, 1, 1, 1), "constant").T.to(torch.int64)
    unfold = nn.Unfold(kernel_size=(2, 2), padding=(1, 1))
    out2 = unfold(img).to(torch.int64)
    print(out1 == out2)

    # test code for convolution
    img = torch.randn(2, 3, 10, 10)
    C_in, C_out = 3, 1
    model = NittaConv2d(C_in, C_out, kernel_size=2)
    print(model(img)[0, :, :, 0])
    """

    # compare out 1
    conv1 = nn.Conv2d(3, 10, (3, 5), stride=(2, 1), padding=(4, 2))
    conv2 = NittaConv2d(3, 10, (3, 5), stride=(2, 1), padding=(4, 2))
    weight = torch.arange(0, 450).reshape(10, 3, 3, 5).to(torch.float32).requires_grad_(True)
    bias = torch.arange(0, 10).to(torch.float32).requires_grad_(True)
    conv1.weight = nn.Parameter(weight)
    conv1.bias = nn.Parameter(bias)
    conv2.weight = nn.Parameter(weight)
    conv2.bias = nn.Parameter(bias)
    img = torch.arange(0, 300).reshape(1, 3, 10, 10).to(torch.float32)
    # print(conv1(img).shape)
    # print(conv1(img)[0, 0, 1, 1])
    # print(conv2(img)[0, 0, 1, 1])
    # print((conv1(img) - conv2(img))[0, 0, :, :])
    print((torch.sum(conv1(img) == conv2(img))))

    # compare out 2
    C_in = 16
    C_out = 33
    kernel_size = (3, 5)
    stride = (2, 1)
    padding = (4, 2)
    conv1 = NittaConv2d(C_in, C_out, kernel_size, stride=stride, padding=padding)
    conv2 = nn.Conv2d(C_in, C_out, kernel_size, stride=stride, padding=padding)
    N = C_out * C_in * kernel_size[0] * kernel_size[1]
    weight = torch.arange(0, N).reshape(C_out, C_in, *kernel_size).to(torch.float32).requires_grad_(True)
    weight = nn.Parameter(weight)
    bias = torch.arange(C_out).to(torch.float32).requires_grad_(True)
    bias = nn.Parameter(bias)
    conv1.weight = weight
    conv2.weight = weight
    conv1.bias = bias
    conv2.bias = bias
    N = C_in * 10 * 10
    img = torch.arange(0, N).reshape(1, C_in, 10, 10).to(torch.float32).requires_grad_(True)
    label = torch.randn(1, C_out, 8, 10)
    # print(conv1(img)[0, 0, :, :])
    # print(conv2(img)[0, 0, :, :])
    # print(conv1(img).shape)
    # print(conv2(img).shape)
    # print(conv1(img) == conv2(img))
    """
    idx = (conv1(img) != conv2(img))
    d1 = conv1(img)[idx]
    d2 = conv2(img)[idx]
    for i in range(len(d1)):
        print(d1[i], d2[i], d1[i] - d2[i])
    """

    criterion = nn.CrossEntropyLoss()
    out1 = conv1(img)
    # print(out1.shape)
    loss1 = criterion(out1, label)
    loss1.backward()
    print(conv1.weight.grad[0, 0, :, :])
    out2 = conv2(img)
    loss2 = criterion(out2, label)
    loss2.backward()
    print(conv2.weight.grad[0, 0, :, :] / 2)
