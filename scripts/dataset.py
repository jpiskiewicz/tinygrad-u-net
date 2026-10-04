#!/usr/bin/env python3

import argparse
from tinygrad_unet.dataset import TRAINING_SIZE, VALIDATION_PART, SOURCE_PATTERNS, TRAIN_DATASET, VAL_DATASET, choose_files, Dataset

if __name__ == "__main__":
  arg_parser = argparse.ArgumentParser(
    prog="Dataset Generator",
    description="Compiles training and validation datasets and saves them as .safetensors files."
  )
  arg_parser.add_argument("--count", "-c", type=int, default=TRAINING_SIZE)
  arg_parser.add_argument("--validation-part", "-p", type=float, default=VALIDATION_PART)
  arg_parser.add_argument("--directory", "-d", type=str)
  args = arg_parser.parse_args()
  train, val = choose_files(SOURCE_PATTERNS, args.count, args.validation_part)
  print("Generating datasets from files:")
  print("Generating training dataset...")
  Dataset(train).save(f"{args.directory}/train_dataset.safetensors" if args.directory else TRAIN_DATASET)
  print("Done.")
  print("Generating validation dataset...")
  Dataset(val).save(f"{args.directory}/val_dataset.safetensors" if args.directory else VAL_DATASET)
  print("Done.")
