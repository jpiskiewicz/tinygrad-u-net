#!/usr/bin/env python3

from tinygrad_unet.dataset import TrivialAugument, load_dataset
from tinygrad_unet.util import make_8bit
from PIL import Image


if __name__ == "__main__":
  dataset = load_dataset("../compiled_datasets/train_dataset_test.safetensors")
  augument = TrivialAugument(dataset)
  augumented_dataset = augument.augument()
  for i in range(10):
    Image.fromarray(make_8bit(dataset[0][i])).save(f"image_original_{i}.png")
    Image.fromarray(make_8bit(dataset[1][i])).save(f"label_original_{i}.png")
    Image.fromarray(make_8bit(augumented_dataset[0][i])).save(f"image_{i}.png")
    Image.fromarray(make_8bit(augumented_dataset[1][i])).save(f"label_{i}.png")