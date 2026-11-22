#!/usr/bin/env python3

import glob
import json
import os
from tinygrad import Variable
from tinygrad.engine.jit import TinyJit
from tinygrad.device import Device
from tinygrad.dtype import dtypes
from tinygrad.nn.state import safe_load, safe_save
from tinygrad.tensor import Tensor
from tinygrad.helpers import tqdm, Context
from tinygrad_unet.util import make_8bit
from tinygrad.uop.ops import sint
from random import shuffle, choice, random
from PIL import Image, ImageFilter
import numpy as np
from concurrent.futures import ProcessPoolExecutor
from abc import ABC, abstractmethod
from typing import cast, Callable, override, final
import math


TRAINING_SIZE = 750
VALIDATION_PART = 0.1
SIZE = 240
TRAIN_DATASET = "compiled_datasets/train/train_dataset.safetensors"
VAL_DATASET = "compiled_datasets/train/val_dataset.safetensors"
FILTER = ImageFilter.MedianFilter(size=21)
SOURCE_PATTERNS = [
    "dataset/benign/*(*).png",
    "dataset/malignant/*(*).png",
    "dataset/normal/*(*).png"
]


def convert_to_device(loaded: dict[str, Tensor]) -> list[Tensor]: return [x.to(Device.DEFAULT).realize() for x in loaded.values()]
def load_dataset(filename: str) -> list[Tensor]: return convert_to_device(safe_load(filename))


def choose_files(patterns:list[str], training_size:int=TRAINING_SIZE, validation_part:float=VALIDATION_PART) -> tuple[list[str], list[str]]:
  validation_size = int(training_size * validation_part)
  files = [x[:-4] for pattern in patterns for x in glob.glob(pattern)]
  shuffle(files)
  train = files[:training_size]
  val = files[training_size:training_size+validation_size]
  print(len(train), len(val))
  assert len(train) > 0 and len(val) > 0
  with open("training_files.json", "w") as f: json.dump(train, f, indent=2)
  with open("validation_files.json", "w") as f: json.dump(val, f, indent=2)
  return train, val


def load_image(p: str) -> Image.Image: return Image.open(p).convert("L")


def make_array(im: Image.Image) -> np.typing.NDArray[np.uint8]: return np.array(im)


def load_image_and_apply_filter(p: str) -> np.typing.NDArray[np.uint8]: return make_array(load_image(p))


def transform_image(p: np.typing.NDArray[np.uint8]) -> np.typing.NDArray[np.float16]:
  im = (p / 255).astype(np.float32) # Normalize
  # Resize the Image to exactly INPUT_SIZE and add mirror padding if the other dimension is < INPUT_SIZE
  aspect = min(im.shape) / max(im.shape)
  desired_size = [round(x) for x in (SIZE if im.shape[0] > im.shape[1] else im.shape[0] + (SIZE - im.shape[1]) * aspect, im.shape[1] + (SIZE - im.shape[0]) * aspect if im.shape[0] > im.shape[1] else SIZE)]
  padding = [(SIZE - x) // 2 for x in desired_size]
  for ax in (1, 0):
    num = np.clip((np.arange(desired_size[ax]) * 2 + 1) * im.shape[ax] - desired_size[ax], 0, (im.shape[ax] - 1) * desired_size[ax] * 2)
    low, high = np.take(im, num // (desired_size[ax] * 2), ax), np.take(im, -(-num // (desired_size[ax] * 2)), ax)
    im = low + (high - low) * np.expand_dims(num % (desired_size[ax] * 2) / (desired_size[ax] * 2), 1 - ax)
  return np.pad(im, ((padding[0] + desired_size[0] % 2, padding[0]), (padding[1] + desired_size[1] % 2, padding[1])), mode="reflect").astype(np.float16)[None, None]


def load_mask(p: list[np.typing.NDArray[np.uint8]]) -> np.typing.NDArray[np.float16]: return transform_image(np.bitwise_or.reduce(p))


def get_masks(p: str) -> list[str]: return glob.glob(p + "_mask*.png")


class Dataset:
  def __init__(self, files: list[str]):
    image_files, label_files = [x + ".png" for x in files], [get_masks(x) for x in files]
    with ProcessPoolExecutor(12) as executor:
      images = tqdm(executor.map(load_image_and_apply_filter, image_files), desc="Loading images", total=len(image_files))
      labels = tqdm(executor.map(self.load_multiple, label_files), desc="Loading labels", total=len(label_files))
    self.images, self.labels = self.load_images(images), self.load_masks(labels)

  def load_multiple(self, paths: list[str]) -> list[np.typing.NDArray[np.uint8]]: return [make_array(load_image(x)) for x in paths]

  def load_images(self, images: list[np.typing.NDArray[np.uint8]]) -> Tensor: return self.combine([transform_image(x) for x in images])

  def load_masks(self, files: list[list[np.typing.NDArray[np.uint8]]]) -> Tensor: return self.combine([load_mask(x) for x in files])

  def combine(self, slices: list[np.typing.NDArray[np.float16]]) -> Tensor: return Tensor(np.stack(slices))

  def save(self, filename: str):
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    safe_save({ "images": self.images, "labels": self.labels }, filename)


class Transform[T](ABC):
  def __init__(self, strength: float):
    self.strength: float = strength

  @abstractmethod
  def apply(self, image: T, mask: T) -> tuple[T, T]: pass


type ImagePair = tuple[Image.Image, Image.Image]
type TensorPair = tuple[Tensor, Tensor]


class ImageTransform(Transform[Image.Image], ABC):
  @override
  @abstractmethod
  def apply(self, image: Image.Image, mask: Image.Image) -> ImagePair: pass


class TensorTransform(Transform[Tensor], ABC):
  @override
  @abstractmethod
  def apply(self, image: Tensor, mask: Tensor) -> TensorPair: pass

@final
class Rotate(ImageTransform):
  def apply_one(self, image: Image.Image) -> Image.Image:
    angle = self.strength * 45
    # Original dimensions
    w, h = image.size

    # Convert angle to radians for math functions
    angle_rad = math.radians(angle)

    # Compute the expanded bounding box size after rotation (for the canvas)
    cos_a = abs(math.cos(angle_rad))
    sin_a = abs(math.sin(angle_rad))
    new_w = int(math.ceil(w * cos_a + h * sin_a))
    new_h = int(math.ceil(w * sin_a + h * cos_a))

    # Perform the rotation with expansion (transparent fill for corners)
    rotated = image.rotate(angle, resample=Image.BICUBIC, expand=True)

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

  @override
  def apply(self, image: Image.Image, mask: Image.Image) -> ImagePair:
    return self.apply_one(image), self.apply_one(mask)


@final
class Zoom(ImageTransform):
  def __init__(self, strength: float):
    super().__init__(strength)
    crop = SIZE * self.strength * 0.25
    self.border = crop // 2

  def apply_one(self, image: Image.Image) -> Image.Image:
    return image.crop((self.border, self.border, SIZE - self.border, SIZE - self.border))

  @override
  def apply(self, image: Image.Image, mask: Image.Image) -> ImagePair: return self.apply_one(image), self.apply_one(mask)


@final
class Translate(ImageTransform):
  def __init__(self, strength: float):
    super().__init__(strength)

    size = SIZE          # since square → width = height
    max_shift = size * 0.25     # 25% of side length

    # Angle: 0° = right, 90° = down, 180° = left, 270° = up
    angle_deg = self.strength * 360
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
    self.crop_box = (
        int(left),           # left
        int(top),            # top
        int(size - right),   # right
        int(size - bottom)   # bottom
    )


  @override
  def apply(self, image: Image.Image, mask: Image.Image) -> ImagePair:
    return image.crop(self.crop_box), mask.crop(self.crop_box)


@final
class SmoothDeform(TensorTransform):
  def __init__(self, strength: float):
    super().__init__(strength)
    KERNEL_SIZE = 3
    # with Context(BEAM=0):
    #   self.kernel = Tensor.normal((KERNEL_SIZE, KERNEL_SIZE, 2), mean=0, std=10 * self.strength, dtype=dtypes.float16).realize()
    self.kernel = Tensor(np.random.normal(loc=0, scale=10 * strength, size=(KERNEL_SIZE, KERNEL_SIZE, 2)).astype(np.float16), device=Device.DEFAULT).realize()

  def source_coords(self, in_size: int, out_size: int) -> Tensor:
      """
      This is a sampling map which pairs the output pixels to source pixels.
      The values can be fractional which means that a given position in the output
      tensor maps to some place in between two pixels in the source tensor.
      """
      return Tensor.arange(out_size, dtype=dtypes.float16).add(0.5).div(out_size/in_size).sub(0.5)

  def cubic_close(self, x: Tensor, a: float) -> Tensor: return ((a + 2) * x - (a + 3)) * x**2 + 1

  def cubic_far(self, x: Tensor, a: float) -> Tensor: return ((a * x - 5 * a) * x + 8 * a) * x - 4 * a

  def cubic_weights(self, t: Tensor, a: float = -0.5) -> tuple[Tensor, ...]:
      return self.cubic_far(t + 1, a), self.cubic_close(t, a), self.cubic_close(1 - t, a), self.cubic_far(2 - t, a)

  def bicubic_conv(self, x: Tensor, i: int, base: Tensor, t: Tensor, weight_dims: tuple[sint, ...], index_dims: tuple[sint, ...]) -> Tensor:
      """
      This performs bicubic convolution in a single dimension.
      """
      weights = self.cubic_weights(t)
      out = Tensor.zeros(index_dims, dtype=dtypes.float16)
      for offset, weight in zip((-1, 0, 1, 2), weights):
          idx = (base + offset).clip(0, x.shape[i] - 1).reshape(weight_dims).expand(index_dims)
          samples = x.gather(i, idx)
          out += samples * weight.reshape(weight_dims).expand(index_dims)
      return out

  def kernel_upscale(self, kernel: Tensor) -> Tensor:
    """
    Perform bicubic interpolation on the kernel in order to bring it up to the provided dimensions.
    """
    x = kernel
    for i in range(2): # Bicubic convolution is a separable operation
        src = self.source_coords(int(kernel.shape[i]), SIZE)
        base = src.floor().cast(dtypes.int)
        t = src - base
        weight_dims = SIZE if i == 0 else 1, SIZE if i == 1 else 1, 1
        index_dims = SIZE, SIZE if i == 1 else kernel.shape[1], 2
        x = self.bicubic_conv(x, i, base, t, weight_dims, index_dims)
    return x

  def construct_sampling_maps(self, kernel: Tensor) -> TensorPair:
    shift_map = self.kernel_upscale(kernel)
    shift_map_x = shift_map[:, :, 0]
    shift_map_y = shift_map[:, :, 1]
    xs = Tensor.arange(SIZE).reshape(1, SIZE).expand(SIZE, SIZE).add(shift_map_x).clip(0, SIZE - 1)
    ys = Tensor.arange(SIZE).reshape(SIZE, 1).expand(SIZE, SIZE).add(shift_map_y).clip(0, SIZE - 1)
    return xs, ys

  def remap_nb(self, image: Tensor, xs: Tensor, ys: Tensor) -> Tensor:
      """
      This remaps the image pixels using nearest-neighbour using the x and y coordinate maps.
      It rounds fractional values so that one source pixel can map to multiple
      output pixels. The opposite is also true - some source pixels can map to
      nothing so they get lost.
      """
      return image[ys.round().cast(dtypes.int), xs.round().cast(dtypes.int)].reshape(1, 1, *image.shape)

  def remap_bicubic(self, image: Tensor, xs: Tensor, ys: Tensor) -> Tensor:
      """
      This takes in the image and two x and y coordinate maps.
      The maps can contain fractional values. Fractions are treated as weights
      for the bicubic interpolation. They define which of the 4 neighbouring pixels
      contribute the most to the value of the current output pixel.
      """
      x0 = xs.floor().cast(dtypes.int)
      y0 = ys.floor().cast(dtypes.int)
      tx = xs - x0
      ty = ys - y0
      wxs = self.cubic_weights(tx)
      wys = self.cubic_weights(ty)

      def sample(ox: int, oy: int) -> Tensor: return image[y0 + oy, x0 + ox]

      row0 = sample(-1, -1) * wxs[0] + sample(0, -1) * wxs[1] + sample(1, -1) * wxs[2] + sample(2, -1) * wxs[3]
      row1 = sample(-1, 0) * wxs[0] + sample(0, 0) * wxs[1] + sample(1, 0) * wxs[2] + sample(2, 0) * wxs[3]
      row2 = sample(-1, 1) * wxs[0] + sample(0, 1) * wxs[1] + sample(1, 1) * wxs[2] + sample(2, 1) * wxs[3]
      row3 = sample(-1, 2) * wxs[0] + sample(0, 2) * wxs[1] + sample(1, 2) * wxs[2] + sample(2, 2) * wxs[3]

      return (row0 * wys[0] + row1 * wys[1] + row2 * wys[2] + row3 * wys[3]).reshape(1, 1, *image.shape)

  @TinyJit
  def smooth_deform(self, kernel: Tensor, image: Tensor, mask: Tensor) -> TensorPair:
    xs, ys = self.construct_sampling_maps(kernel)
    return self.remap_bicubic(image, xs, ys), self.remap_nb(mask, xs, ys)

  @override
  def apply(self, image: Tensor, mask: Tensor) -> TensorPair:
     # tinygrad wraps jitted functions into a getter method which doesn't have any typing set up.
     # That's why we need to explicitly cast the return value of smooth_deform.
     # https://github.com/tinygrad/tinygrad/blob/master/tinygrad/engine/jit.py?utm_source=chatgpt.com#L120
    return cast(TensorPair, self.smooth_deform(self.kernel, image, mask))


class TrivialAugument:
  """
  This class is to be used in a separate script than the one that contains the training routine.
  During training this class will generate new transforms of the training dataset which will be
  saved in new .safetensors files which then will be ingested by the training routine.
  The whole thing is to run in a separate thread so that it doesn't slow down the training process.
  """
  def __init__(self, dataset: list[Tensor], transform_strength: Callable[[], float] = random):
    # Convert tensors to PIL images
    self.image_tensors, self.label_tensors = dataset[0], dataset[1]
    self.images, self.labels = [Image.fromarray(make_8bit(x)) for x in self.image_tensors], [Image.fromarray(make_8bit(x)) for x in self.label_tensors]
    self.transformations = [Rotate, Zoom, Translate, SmoothDeform]
    self.transform_strength = transform_strength

  def convert_img_to_out_tensor(self, x: Image.Image) -> Tensor:
    return (Tensor(make_array(x)).cast(dtypes.float16) / 255).reshape(1, 1, SIZE, SIZE)

  def apply_image(self, img: Image.Image, mask: Image.Image, transform: ImageTransform) -> TensorPair:
    transformed_img, transformed_mask = transform.apply(img, mask)
    transformed_img = transformed_img.resize((SIZE, SIZE), Image.Resampling.BICUBIC)
    transformed_mask = transformed_mask.resize((SIZE, SIZE), Image.Resampling.NEAREST)
    out = self.convert_img_to_out_tensor(transformed_img), self.convert_img_to_out_tensor(transformed_mask)
    return cast(TensorPair, out)

  def apply_tensor(self, img: Tensor, mask: Tensor, transform: TensorTransform) -> TensorPair: return transform.apply(img, mask)
    
  @TinyJit
  def extract_pair(self, images: Tensor, masks: Tensor, index: Tensor) -> TensorPair:
    image, mask = images[index].reshape(SIZE, SIZE).contiguous(), masks[index].reshape(SIZE, SIZE).contiguous()
    image.realize(mask)
    return image, mask
    
  def get_tensor_index(self, i: int) -> Tensor: return Tensor([i], dtype=dtypes.int, device=Device.DEFAULT).realize()

  def run_transform(self, i: int, transform: ImageTransform | TensorTransform) -> TensorPair:
    if isinstance(transform, ImageTransform): return self.apply_image(self.images[i], self.labels[i], transform)
    tensor_index = self.get_tensor_index(i)
    image, mask = cast(TensorPair, self.extract_pair(self.image_tensors, self.label_tensors, tensor_index))
    return self.apply_tensor(image, mask, transform)
    
  def augument(self) -> list[Tensor]:
    print("Performing TrivialAugument on the dataset...")
    count = len(self.images)
    shape = (count, 1, 1, SIZE, SIZE)
    images, labels = Tensor.empty(shape, dtype=dtypes.float16).realize(), Tensor.empty(shape, dtype=dtypes.float16).realize()
    for i in range(count):
      transform = choice(self.transformations)(self.transform_strength())
      image, label = self.run_transform(i, transform)
      image.realize(label)
      tensor_index = Variable("tensor_index", 0, count - 1).bind(i)
      images[tensor_index].assign(image)
      labels[tensor_index].assign(label)
      images.realize(labels)
    return [images, labels]
      