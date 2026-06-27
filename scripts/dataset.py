#!/usr/bin/env python3

from tinygrad_unet.dataset import SOURCE_PATTERNS, TRAIN_DATASET, VAL_DATASET, choose_files, Dataset

if __name__ == "__main__":
   train, val = choose_files(SOURCE_PATTERNS)
   print("Generating datasets from files:")
   print("Generating training dataset...")
   Dataset(train).save(TRAIN_DATASET)
   print("Done.")
   print("Generating validation dataset...")
   Dataset(val).save(VAL_DATASET)
   print("Done.")
