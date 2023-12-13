from torch import nn


class Describer(nn.Module):
    def __init__(self):
        ...

    def forward(self, x):
        descriptors = None

        return descriptors


class Translator(nn.Module):
    def __init__(self):
        ...

    def forward(self, descriptor_a, descriptor_b):
        transformation = None
        return transformation
