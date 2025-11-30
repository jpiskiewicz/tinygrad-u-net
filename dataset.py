#!/usr/bin/env python3

import random
import glob
import json

from tinygrad.engine.jit import TinyJit
from tinygrad.tensor import Tensor
from tinygrad.helpers import tqdm
from helpers import plot_slice
from PIL import Image
import numpy as np
from concurrent.futures import ProcessPoolExecutor


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
  
  
def load_image(p: str) -> np.typing.NDArray: return np.array(Image.open(p).convert("L"), np.float16)
  
  
def transform_image(p: np.typing.NDArray) -> Tensor:
  # Resize the Image to exactly INPUT_SIZE and add mirror padding if the other dimension is < INPUT_SIZE
  im = Tensor(p) / 255
  aspect = min(im.shape) / max(im.shape)
  desired_size = [round(x) for x in (SIZE if im.shape[0] > im.shape[1] else im.shape[0] + (SIZE - im.shape[1]) * aspect, im.shape[1] + (SIZE - im.shape[0]) * aspect if im.shape[0] > im.shape[1] else SIZE)]
  padding = [(SIZE - x) // 2 for x in desired_size]
  im = im.interpolate(desired_size).pad((padding[1] + desired_size[1] % 2, padding[1], padding[0] + desired_size[0] % 2, padding[0]), mode = "reflect").expand(1, 1, -1, -1)
  return im
  
  
def load_mask(p: list[np.typing.NDArray]) -> Tensor:
  masks = [transform_image(x) for x in p]
  combined = masks[0]
  for mask in masks[1:]: combined += mask
  return combined
  

def get_masks(p: str) -> list[str]: return glob.glob(p + "_mask*.png")


class Dataset:
  def __init__(self, files: list[str]):
    image_files, label_files = [x + ".png" for x in files], [get_masks(x) for x in files]
    with ProcessPoolExecutor(12) as executor:
      images = tqdm(executor.map(load_image, image_files), desc="Loading images", total=len(image_files))
      labels = tqdm(executor.map(self.load_multiple, label_files), desc="Loading labels", total=len(label_files))
    self.images, self.labels = self.load_images(images), self.load_masks(labels)
  
  def load_multiple(self, paths: list[str]) -> list[np.typing.NDArray]: return [load_image(x) for x in paths]
  
  def load_images(self, images: list[np.typing.NDArray]): return self.combine([transform_image(x) for x in images])
  
  def load_masks(self, files: list[list[np.typing.NDArray]]): return self.combine([load_mask(x) for x in files])

  @TinyJit
  def combine(self, slices: list[Tensor]) -> Tensor: return slices[0].stack(*slices[1:]).realize()


if __name__ == "__main__":
   train, val = choose_files(REGEX)
   print("Creating dataset")
   dataset = Dataset(val)
   print("Creating dataset done")
   for i in range(len(dataset.labels)):
     plot_slice(dataset.labels[i][0][0])
     plot_slice(dataset.images[i][0][0])
