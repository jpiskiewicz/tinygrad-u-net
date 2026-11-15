#!/usr/bin/env python3

"""
This script runs in the background at the same time the train script runs
and performs TrivialAugument on the input dataset.
It saves it then in the location from which the train script picks it up.
"""

from tinygrad_unet.dataset import TRAIN_DATASET, load_dataset, TrivialAugument
from tinygrad.nn.state import safe_save
from argparse import ArgumentParser

if __name__ == "__main__":
  argument_parser = ArgumentParser(
    prog="Dataset augumentation script",
    description="This script runs a randomly chosen transform with random strength on every image - label pair in the dataset."
  )
  argument_parser.add_argument("-d", "--dataset-location", default=TRAIN_DATASET)
  argument_parser.add_argument("-s", "--start-epoch", type=int, default=2)
  argument_parser.add_argument("epochs", type=int)
  args = argument_parser.parse_args()
  dataset = TrivialAugument(load_dataset(args.dataset_location))
  epochs = int(args.epochs)
  for i in range(args.start_epoch, epochs + 1):
    images, labels = dataset.augument()
    safe_save({ "images": images, "labels": labels }, f"{args.dataset_location.split('.')[0]}_{i}.safetensors")
  print(f"TrivialAugument script completed. Dataset is ready for {epochs} epochs.")
