import torch
from torch import nn
from torch import cuda

# import flyingthings3d dataset

from torchvision.datasets import CarlaStereo


class Network(nn.Module):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def forward(self, input, output):
        return torch.add(input, input, out=output)


def main():
    things = CarlaStereo
    stream = cuda.Stream()
    cugraph = cuda.CUDAGraph()
    graph = cuda.graph(cugraph, stream=stream)

    image = torch.ones((16, 16), dtype=torch.float16, device="cuda")
    image_out = torch.empty_like(image)
    with graph:
        image_out = torch.add(image, image, out=image_out)
    # with graph:
    #     torch.add(image, image, out=image)
    # network = Network()
    # graph = cuda.make_graphed_callables(network, sample_args=(image, image_out))
    cugraph.replay()
    a = 1


if __name__ == "__main__":
    main()
