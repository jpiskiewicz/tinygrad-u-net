#!/usr/bin/env python3

"""
This script runs in the background at the same time the train script runs
and performs TrivialAugument on the input dataset.
It saves it then in the location from which the train script picks it up.
"""

from tinygrad_unet.dataset import TRAIN_DATASET, load_dataset, TrivialAugument
from tinygrad.nn.state import safe_save
from sys import argv


if __name__ == "__main__":
  dataset = TrivialAugument(load_dataset(TRAIN_DATASET))
  for i in range(1, int(argv[1])): 
    images, labels = dataset.augument()
    safe_save({ "images": images, "labels": labels }, f"{TRAIN_DATASET.split('.')[0]}_{i}.safetensors")
    