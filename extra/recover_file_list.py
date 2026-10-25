#!/usr/bin/env python3

"""
This script takes a .safetensors file as input and a path to the dataset directory.
It then converts all non-mask files in that directory using the same logic as when
compiling the dataset.
Then it filters the converted files and returns a list of paths to files in the dataset
that contain the same pixels as images in the input .safetensors file.
"""

import json
from argparse import ArgumentParser
from tinygrad.nn.state import safe_load
from tinygrad.tensor import Tensor
from glob import glob
from tinygrad_unet.dataset import convert_to_device, load_image_and_apply_filter, transform_image


type ImageSet = list[tuple[str, Tensor]]


def get_all_images(path: str) -> ImageSet:
  files = glob(f"{path}/**/*(*).png")
  return list(zip(files, [transform_image(load_image_and_apply_filter(x)).realize() for x in files]))


def get_matching_paths(dataset: Tensor, chosen_image_set: ImageSet) -> list[str]: 
  return [pair[0] for pair in filter(lambda x: any((y == x[1]).all().item() for y in dataset), chosen_image_set)]


if __name__ == "__main__":
  arg_parser = ArgumentParser(prog="Recover File List")
  arg_parser.add_argument("dataset")
  arg_parser.add_argument("image_dir")
  args = arg_parser.parse_args()
  print("Loading dataset .safetensors file...")
  dataset = convert_to_device(safe_load(args.dataset))[0]
  print("Done.")
  print("Loading and converting image files into tensors...")
  chosen_image_set = get_all_images(args.image_dir)
  print("Done.")
  print("Getting matches...")
  pairs = get_matching_paths(dataset, chosen_image_set)
  print("Done.")
  print(json.dumps(pairs, indent=2))