#!/usr/bin/env python3

"""
The idea of these tests is to run all of the prerequisites for training
and then run the training itself to see whether all stages run successfully.
The stages that this script is testing are:
1. dataset preparation;
2. dataset augumentation;
3. training.
"""

import unittest
import shutil
from tinygrad_unet.dataset import SOURCE_PATTERNS, Dataset, TrivialAugument, choose_files, load_dataset
from scripts.train import run_training
from tinygrad.nn.state import safe_save
from typing import final
from pathlib import Path

TESTFILE_DIR = Path(__file__).resolve().parent
DATASET_DIR = TESTFILE_DIR / "compiled_datasets"

@final
class TestTrain(unittest.TestCase):
  TRAIN_DATASET_LOCATION = f"{DATASET_DIR}/train_dataset.safetensors"
  VAL_DATASET_LOCATION = f"{DATASET_DIR}/val_dataset.safetensors"
  TRAIN_DATASET_TENSOR_SHAPE = (10, 1, 1, 240, 240)
  
  def test_dataset(self):
    train, val = choose_files(SOURCE_PATTERNS)
    train = train[:10]
    val = val[:1]
    Dataset(train).save(self.TRAIN_DATASET_LOCATION)
    train_images, train_labels = load_dataset(self.TRAIN_DATASET_LOCATION)
    self.assertEqual(train_images.shape, self.TRAIN_DATASET_TENSOR_SHAPE)
    self.assertEqual(train_labels.shape, self.TRAIN_DATASET_TENSOR_SHAPE)
    Dataset(val).save(self.VAL_DATASET_LOCATION)
    val_images, val_labels = load_dataset(self.VAL_DATASET_LOCATION)
    VAL_DATASET_TENSOR_SHAPE = (1, 1, 1, 240, 240)
    self.assertEqual(val_images.shape, VAL_DATASET_TENSOR_SHAPE)
    self.assertEqual(val_labels.shape, VAL_DATASET_TENSOR_SHAPE)
    
  def test_augument(self):
    dataset = TrivialAugument(load_dataset(self.TRAIN_DATASET_LOCATION))
    images_augumented, labels_augumented = dataset.augument()
    augumented_location = f"{self.TRAIN_DATASET_LOCATION.split('.')[0]}_2.safetensors"
    safe_save({ "images": images_augumented, "labels": labels_augumented }, augumented_location)
    loaded_images, loaded_labels = load_dataset(augumented_location)
    self.assertEqual(loaded_images.shape, self.TRAIN_DATASET_TENSOR_SHAPE)
    self.assertEqual(loaded_labels.shape, self.TRAIN_DATASET_TENSOR_SHAPE)
    
  def test_train(self):
    # This test isn't opinionated in any way. It just checks whether the training routine goes through without crashing.
    run_training(load_dataset(self.VAL_DATASET_LOCATION), 2, None, str(TESTFILE_DIR / "predictions"), self.TRAIN_DATASET_LOCATION)
    
    
def ordering_in_test_case(name: str) -> int: return list(TestTrain.__dict__).index(name)
    
if __name__ == "__main__":
  # Clean the directory with
  for path in DATASET_DIR.iterdir():
    if path.is_dir(): shutil.rmtree(path)
    else: path.unlink()
  unittest.defaultTestLoader.sortTestMethodsUsing = lambda a, b: ordering_in_test_case(a) - ordering_in_test_case(b)
  unittest.main()
