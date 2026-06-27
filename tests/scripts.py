#!/usr/bin/env python3

"""
The idea of these tests is to run all of the core scripts in their proper order
and see whether all stages run successfully.
The stages mentioned here are:
1. dataset preparation;
2. dataset augumentation;
3. training.
"""

import unittest
from tinygrad_unet.dataset import SOURCE_PATTERNS, Dataset, choose_files, load_dataset

class TestScripts(unittest.TestCase):
  def test_dataset(self):
    train, val = choose_files(SOURCE_PATTERNS)
    train = train[:10]
    val = val[:1]
    TRAIN_DATASET_LOCATION = "tests/compiled_datasets/train_dataset.safetensors"
    Dataset(train).save(TRAIN_DATASET_LOCATION)
    train_images, train_labels = load_dataset(TRAIN_DATASET_LOCATION)
    TRAIN_DATASET_TENSOR_SHAPE = (10, 1, 1, 240, 240)
    self.assertEqual(train_images.shape, TRAIN_DATASET_TENSOR_SHAPE)
    self.assertEqual(train_labels.shape, TRAIN_DATASET_TENSOR_SHAPE)
    VAL_DATASET_LOCATION = "tests/compiled_datasets/val_dataset.safetensors"
    Dataset(val).save(VAL_DATASET_LOCATION)
    val_images, val_labels = load_dataset(VAL_DATASET_LOCATION)
    VAL_DATASET_TENSOR_SHAPE = (1, 1, 1, 240, 240)
    self.assertEqual(val_images.shape, VAL_DATASET_TENSOR_SHAPE)
    self.assertEqual(val_labels.shape, VAL_DATASET_TENSOR_SHAPE)
    
if __name__ == "__main__":
  unittest.main()