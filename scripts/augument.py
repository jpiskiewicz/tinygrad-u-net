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
  epochs = int(argv[1])
  for i in range(2 if len(argv) == 2 else int(argv[2]), epochs + 1): 
    images, labels = dataset.augument()
    safe_save({ "images": images, "labels": labels }, f"{TRAIN_DATASET.split('.')[0]}_{i}.safetensors")
  print(f"TrivialAugument script completed. Dataset is ready for {epochs} epochs.")
    