#!/usr/bin/env python3

import glob
import json
# from tinygrad.engine.jit import TinyJit
from tinygrad.nn.state import safe_load, safe_save
from tinygrad.tensor import Tensor
from tinygrad.helpers import tqdm
from tinygrad_unet.util import make_8bit
from random import shuffle, choice, random
from PIL import Image, ImageFilter
import numpy as np
from concurrent.futures import ProcessPoolExecutor
from typing import Callable
import math


TRAINING_SIZE = 10
VALIDATION_PART = 0.1
SIZE = 240
TRAIN_DATASET = "compiled_datasets/train_dataset_test.safetensors"
VAL_DATASET = "compiled_datasets/val_dataset_test.safetensors"
FILTER = ImageFilter.MedianFilter(size=21)


def convert_to_device(loaded: dict[str, Tensor]) -> list[Tensor]: return [x.to("AMD") for x in loaded.values()]
def load_dataset(filename: str) -> list[Tensor]: return convert_to_device(safe_load(filename))


def choose_files(patterns):
  validation_size = int(TRAINING_SIZE * VALIDATION_PART)
  files = [x[:-4] for pattern in patterns for x in glob.glob(pattern)]
  shuffle(files)
  train = files[:TRAINING_SIZE]
  val = files[TRAINING_SIZE:TRAINING_SIZE+validation_size]
  print(len(train), len(val))
  assert len(train) > 0 and len(val) > 0
  with open("training_files.json", "w") as f: json.dump(train, f, indent=2)
  with open("validation_files.json", "w") as f: json.dump(val, f, indent=2)
  return train, val
  
  
def load_image(p: str) -> Image.Image: return Image.open(p).convert("L")


def make_array(im: Image.Image) -> np.typing.NDArray: return np.array(im, np.float16)


def load_image_and_apply_filter(p: str) -> np.typing.NDArray: return make_array(load_image(p))
  
  
def transform_image(p: np.typing.NDArray) -> Tensor:
  im = Tensor(p) / 255 # Normalize
  # Resize the Image to exactly INPUT_SIZE and add mirror padding if the other dimension is < INPUT_SIZE
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
      images = tqdm(executor.map(load_image_and_apply_filter, image_files), desc="Loading images", total=len(image_files))
      labels = tqdm(executor.map(self.load_multiple, label_files), desc="Loading labels", total=len(label_files))
    self.images, self.labels = self.load_images(images), self.load_masks(labels)
  
  def load_multiple(self, paths: list[str]) -> list[np.typing.NDArray]: return [make_array(load_image(x)) for x in paths]
  
  def load_images(self, images: list[np.typing.NDArray]) -> Tensor: return self.combine([transform_image(x) for x in images])
  
  def load_masks(self, files: list[list[np.typing.NDArray]]) -> Tensor: return self.combine([load_mask(x) for x in files])

  # TODO: Jitting this makes it crash in the latest tinygrad. We can uncomment this once this gets fixed. 
  # @TinyJit
  def combine(self, slices: list[Tensor]) -> Tensor: return slices[0].stack(*slices[1:]).realize()
  
  def save(self, filename: str): safe_save({ "images": self.images, "labels": self.labels }, filename)
  
  
type Transform = Callable[[Image.Image], Image.Image]
  
class TrivialAugument:
  """
  This class is to be used in a separate script than the one that contains the training routine.
  During training this class will generate new transforms of the training dataset which will be
  saved in new .safetensors files which then will be ingested by the training routine.
  The whole thing is to run in a separate thread so that it doesn't slow down the training process.
  """
  def __init__(self, dataset: list[Tensor]):
    # Convert tensors to PIL images,
    self.images, self.labels = [Image.fromarray(make_8bit(x)) for x in dataset[0]], [Image.fromarray(make_8bit(x)) for x in dataset[1]]
    self.transformations = [self.rotate, self.zoom, self.translate]
   
  def rotate(self, s: float) -> Transform:
      print(f"rotate({s})")
      def f(img: Image.Image) -> Image.Image:
        angle = s * 45
        # Original dimensions
        w, h = img.size
        
        # Convert angle to radians for math functions
        angle_rad = math.radians(angle)
        
        # Compute the expanded bounding box size after rotation (for the canvas)
        cos_a = abs(math.cos(angle_rad))
        sin_a = abs(math.sin(angle_rad))
        new_w = int(math.ceil(w * cos_a + h * sin_a))
        new_h = int(math.ceil(w * sin_a + h * cos_a))
        
        # Perform the rotation with expansion (transparent fill for corners)
        rotated = img.rotate(angle, resample=Image.BICUBIC, expand=True)
        
        width_is_longer = w >= h
        side_long, side_short = (w, h) if width_is_longer else (h, w)
        
        sin_a_val, cos_a_val = abs(math.sin(angle_rad)), abs(math.cos(angle_rad))
        
        if side_short <= 2.0 * sin_a_val * cos_a_val * side_long or abs(sin_a_val - cos_a_val) < 1e-10:
            # Half-constrained: two crop corners touch the longer side
            x = 0.5 * side_short
            wr, hr = (x / sin_a_val, x / cos_a_val) if width_is_longer else (x / cos_a_val, x / sin_a_val)
        else:
            # Fully-constrained: crop touches all 4 sides
            cos_2a = cos_a_val * cos_a_val - sin_a_val * sin_a_val
            wr = (w * cos_a_val - h * sin_a_val) / cos_2a
            hr = (h * cos_a_val - w * sin_a_val) / cos_2a
        
        # Ensure positive dimensions and round to int
        wr = max(0, int(math.floor(wr)))
        hr = max(0, int(math.floor(hr)))
        
        # Crop from the center of the rotated image
        left = (new_w - wr) // 2
        top = (new_h - hr) // 2
        right = left + wr
        bottom = top + hr
        
        cropped = rotated.crop((left, top, right, bottom))
        
        return cropped
      return f
    
  def zoom(self, s: float) -> Transform:
    print(f"zoom({s})")
    def f(img: Image.Image) -> Image.Image:
      # TODO: Maybe add support for zooming out and reflect?
      crop = SIZE * s * 0.25
      border = crop // 2
      return img.crop((border, border, SIZE - border, SIZE - border))
    return f
    
  def translate(self, s: float) -> Transform:
    print(f"translate({s})")
    def f(img: Image.Image) -> Image.Image:
      size = img.size[0]          # since square → width = height
      max_shift = size * 0.25     # 25% of side length
  
      # Angle: 0° = right, 90° = down, 180° = left, 270° = up
      angle_deg = s * 360
      angle_rad = math.radians(angle_deg)
  
      # Displacement vector (positive = content moves in that direction)
      dx = math.cos(angle_rad) * max_shift   # x: positive = right
      dy = math.sin(angle_rad) * max_shift   # y: positive = down
  
      # Crop amounts (we crop opposite to movement direction)
      left   = max(0,  dx)     # crop left   when moving content right
      right  = max(0, -dx)     # crop right  when moving content left
      top    = max(0,  dy)     # crop top    when moving content down
      bottom = max(0, -dy)     # crop bottom when moving content up
  
      # Create crop box (all values are safe since max_shift = 0.25×size)
      crop_box = (
          int(left),           # left
          int(top),            # top
          int(size - right),   # right
          int(size - bottom)   # bottom
      )
      return img.crop(crop_box)
    return f
    
  # def elastic_deformation(self, img: Image.Image, s: float) -> Image.Image:
  #   pass
    
  def apply(self, img: Image.Image, transform: Transform) -> Tensor: return (Tensor(make_array(transform(img))) / 255).interpolate((SIZE, SIZE), "nearest-exact").expand(1, 1, -1, -1)
  
  def run_transform(self, image: Image.Image, label: Image.Image) -> tuple[Tensor, Tensor]:
    transform = choice(self.transformations)(random())
    return self.apply(image, transform), self.apply(label, transform)
    
  def augument(self) -> list[Tensor]:
    print("Performing TrivialAugument on the dataset...")
    images: list[Tensor] = []
    labels: list[Tensor] = []
    for i in range(len(self.images)):
      image, label = self.run_transform(self.images[i], self.labels[i])
      images.append(image)
      labels.append(label)
    return [images[0].stack(*images[1:]).realize(), labels[0].stack(*labels[1:]).realize()]
  