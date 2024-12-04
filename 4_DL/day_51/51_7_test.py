import torch


def num_gpus():
    return torch.cuda.device_count()
print(num_gpus())