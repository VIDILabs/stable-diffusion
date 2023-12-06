import torch
import torchvision.transforms.functional as tvfun


class Identity(torch.nn.Module):
    def __call__(self, input):
        return input


class Invert(torch.nn.Module):
    def __call__(self, input):
        return tvfun.invert(input)
