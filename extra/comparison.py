#!/usr/bin/env python3

import numpy as np
from tinygrad.nn.state import safe_load
from tinygrad.tensor import Tensor

COMPARISON_SETS = ["benign" , "simulated"]

def convert_to_device(loaded: dict[str, Tensor]) -> list[Tensor]: return [x.to("AMD") for x in loaded.values()]

def load_dataset(filename: str) -> list[Tensor]: return convert_to_device(safe_load(filename))

def mode_and_count(t: Tensor) -> tuple[float, int]:
    # Returns the most common value and overall count of unique values
    arr = t.numpy().flatten()
    values, counts = np.unique(arr, return_counts=True)
    most_common_idx = counts.argmax()
    return values[most_common_idx], len(values)

def print_statistics(image: Tensor, name):
    print(f"Statistics for {name}:")
    print(f"shape = {image.shape}")
    print(f"min = {image.min().numpy()}")
    print(f"max = {image.max().numpy()}")
    print(f"mean = {image.mean().numpy()}")
    print(f"std = {image.std().numpy()}")
    mode, count = mode_and_count(image)
    print(f"mode = {mode}")
    print(f"count of values = {count}")


if __name__ == "__main__":
  for name in COMPARISON_SETS:
    dataset = load_dataset(f"../train_dataset_{name}.safetensors")
    print_statistics(dataset[0][0], f"image {name}")
    print_statistics(dataset[1][0], f"label {name}")