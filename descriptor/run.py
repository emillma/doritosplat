import torch
from torchvision.datasets import Kitti
from torchvision import transforms
import time
from pickle import dumps, loads

import multiprocessing as mp
import random


def fill_with_ones(to_fill_queue: mp.Queue, filled_queue: mp.Queue):
    loading_stream = torch.cuda.Stream()
    i = 0
    while True:
        key, tensor = to_fill_queue.get()

        with torch.cuda.stream(loading_stream):
            tensor.fill_(i)
        filled_queue.put(key)
        i += 1


if __name__ == "__main__":
    mp.set_start_method("spawn")
    images = torch.zeros((100, 3, 256, 256), dtype=torch.float16, device="cuda")
    loading_stream = torch.cuda.Stream()

    to_fill_queue = mp.Queue()
    filled_queue = mp.Queue()
    proc = mp.Process(target=fill_with_ones, args=(to_fill_queue, filled_queue))
    proc.start()
    waiting = {}
    for i in range(10000):
        to_fill_queue.put((i, images))
        waiting[i] = images
        ready = waiting.pop(filled_queue.get())

    proc.terminate()
    here = True
