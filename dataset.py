#!/usr/bin/env python3

import random
import glob
import json
from tinygrad.tensor import Tensor
from helpers import plot_slice
from PIL import Image
import numpy as np


SLICE = 120
SIZE = 240
REGEX = "dataset/**/*(*).png"


def choose_files(pattern):
  files = [x[:-4] for x in glob.glob(pattern)]
  random.shuffle(files)
  idx = int(len(files) * 0.7)
  train = files[:idx]
  val = files[idx:]
  with open("training_files.json", "w") as f: json.dump(train, f, indent=2)
  with open("validation_files.json", "w") as f: json.dump(val, f, indent=2)
  return train, val


def load_image(p: str) -> Tensor:
  # Resize the Image to exactly INPUT_SIZE and add mirror padding if the other dimension is < INPUT_SIZE
  im = Tensor(np.array(Image.open(p).convert("L"), np.float16)) / 255
  aspect = min(im.shape) / max(im.shape)
  desired_size = [round(x) for x in (SIZE if im.shape[0] > im.shape[1] else im.shape[0] + (SIZE - im.shape[1]) * aspect, im.shape[1] + (SIZE - im.shape[0]) * aspect if im.shape[0] > im.shape[1] else SIZE)]
  padding = [(SIZE - x) // 2 for x in desired_size]
  im = im.interpolate(desired_size).pad((padding[1] + desired_size[1] % 2, padding[1], padding[0] + desired_size[0] % 2, padding[0]), mode = "reflect").expand(1, 1, -1, -1)
  print(im.shape)
  return im
  
  
def load_mask(p: str) -> Tensor:
  masks = [load_image(x) for x in glob.glob(p + "_mask*.png")]
  combined = masks[0]
  for mask in masks[1:]: combined += mask
  return combined


class Dataset:
  def __init__(self, files: list[str]): self.images, self.labels = self.load_images(files), self.load_masks(files)
  
  def load_images(self, files: list[str]): return self.combine([load_image(x + ".png") for x in files])
  
  def load_masks(self, files: list[str]): return self.combine([load_mask(x) for x in files])

  def combine(self, slices: list[Tensor]) -> Tensor: return slices[0].stack(*slices[1:]).realize()


if __name__ == "__main__":
   train, val = choose_files(REGEX)
   dataset = Dataset(val)
   for i in range(len(dataset.labels)):
     x = dataset.labels[i][0][0]
     print(x.min().numpy(), x.std().numpy(), x.max().numpy())
     plot_slice(x)
