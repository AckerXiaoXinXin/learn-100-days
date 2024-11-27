import torch


if __name__ == '__main__':
    x = torch.arange(12, dtype=torch.float32)
    print(x)
    print(x.numel())
    print(x.shape(12))

    y = torch.arange(24, dtype=torch.float)
    print(y)
    print(y.numel())
    print(x.shape(12))
    print(y.shape)
    print(x.item())
    print(y.item())
    print(x.type())