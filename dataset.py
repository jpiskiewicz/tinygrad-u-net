#!/usr/bin/env python3

import random
import glob
import json
import nibabel as nib
from os import path
from pathlib import Path
from tinygrad.tensor import Tensor
from helpers import plot_slice
import numpy as np

# For now the idea is to train the net on the 120th slice of each saggital view of the MRI scan. 
# I think that if we figure out how to train the net on such a simple scenario, then figuring out
# how to train it for other (more complex use-cases) will be easier.
 
SLICE = 120 
INPUT_SIZE = 240


def choose_files(pattern):
  directories = glob.glob(pattern)
  random.shuffle(directories)
  idx = int(len(directories) * 0.7)
  train = directories[:idx]
  val = directories[idx:]
  with open("validation_files.json", "w") as f: json.dump(val, f)
  return train, val


class Dataset:
  def __init__(self, directories: list[str]): self.images, self.labels = self.get_slices(self.get_files(directories))
    
  def get_files(self, directories: list[str]) -> list[list[str]]: return [[path.join(dirname, Path(dirname).name + "_" + x + ".nii") for dirname in directories] for x in ["t1ce", "seg"]]
  
  def get_slices(self, data: list[list[str]]) -> list[Tensor]: return [self.combine([self.load_slice(x, i == 1) for x in fileset]) for i, fileset in enumerate(data)]
  
  def load_slice(self, path: str, mask: bool) -> Tensor:
    s = Tensor(nib.load(path).get_fdata(dtype=np.float16)[SLICE]).flip(1).transpose().clamp(0, 1 if mask else None)
    not_even = [s.shape[i] % 2 for i in range(2)]
    [y_pad, x_pad] = [(INPUT_SIZE - s.shape[i]) // 2 for i in range(2)]
    return s.pad((x_pad, x_pad + (1 if not_even[1] else 0), y_pad, y_pad + (1 if not_even[0] else 0))).expand(1, 1, -1, -1)
  
  def combine(self, slices: list[Tensor]) -> Tensor: return slices[0].stack(*slices[1:])
  
  
if __name__ == "__main__":
   train, val = choose_files("dataset/MICCAI_BraTS_2019_Data_Training/*GG/*")
   dataset = Dataset(val)
   for i in range(len(dataset.labels)):
     print(dataset.labels[i].shape)
     plot_slice(dataset.labels[i])
