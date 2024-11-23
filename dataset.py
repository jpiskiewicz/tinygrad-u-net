#!/usr/bin/env python3

import numpy
import cv2
import glob
import re
from itertools import chain
from functools import reduce
from tinygrad.tensor import Tensor
from PIL import Image


class Dataset:
  def __init__(self):
    """
    Dataset build process:
      1. Read masks and images into tensors;
      2. Combine multiple masks for into one object ID channel;
      3. Zip the image and mask tensors together;
      4. Perform smooth image deformations on each tensor;
      5. Combine all deformed tensors.
    """
    dirs = ["benign", "malignant", "normal"]
    image_filenames, mask_filenames = self.get_filenames(dirs)
    images, masks = { x: self.read_image(x) for x in image_filenames }, { x: self.read_image(x, "1") for x in mask_filenames }
    masks = self.combine_masks(masks)
    assert len(images) == len(masks), f"len(images) = {len(images)}, len(masks) = {len(masks)}"

  def collect_glob(self, dirs: list[str], is_mask: bool = False) -> chain[str]:
    return chain.from_iterable(glob.glob(f"data/{x}/*){'_*' if is_mask else ''}.png") for x in dirs)

  def get_filenames(self, dirs: list[str]) -> tuple[list[str], list[str]]: return sorted(self.collect_glob(dirs)), sorted(self.collect_glob(dirs, True))

  def read_image(self, filename: str, mode: str = "L") -> Tensor:
    image = Image.open(filename).convert(mode)
    center = [x/2 for x in image.size]
    cropped = numpy.array(image.crop((center[0] - 286, center[1] - 286, center[0] + 286, center[1] + 286))).astype(numpy.uint8)
    return Tensor(cropped)

  def group_masks(self, masks: dict[str, Tensor]) -> list[list[Tensor]]:
    """
    Groups masks that belong to the same image and combines them together
    into a single object ID channel.
    """
    grouped_masks = []
    last_id: int = -1
    curr_mask = []
    for k, v in masks.items():
      curr_id = int(re.split(r"[()]", k)[1])
      if curr_id != last_id and last_id != -1:
        grouped_masks.append(curr_mask)
        curr_mask = []
      curr_mask.append(v)
      last_id = curr_id
    print(grouped_masks[-1], curr_mask) # TODO)) Maybe append last curr_mask to grouped_masks
    return grouped_masks

  def combine_masks(self, masks: dict[str, Tensor]) -> list[Tensor]: return [reduce(lambda v, e: v + e, x) for x in self.group_masks(masks)] # Fuuck not the Tensor | MathTrait bullshit again...

  def deform(self, image: numpy.ndarray, mask: numpy.ndarray) -> tuple[Tensor, Tensor]:
    """
    This contains the logic for smooth image deformation (https://en.wikipedia.org/wiki/Homotopy)
    using random displacement vectors on a coarse 3x3 grid (see section 3.1 in the U-Net paper).

    Rewrite this to be just one big operation on a multidimensional Tensor.
    """
    grid_size = 3
    sd = 10
    height, width = image.shape

    # Create displacement vectors
    dx, dy = [numpy.random.normal(0, sd,  (grid_size, grid_size)) for _ in range(2)]

    # Create fine meshgrid for the image
    x_fine, y_fine = numpy.meshgrid(numpy.arange(width), numpy.arange(height))

    # Perform bicubic interpolation on displacement vectors to get per-pixel displacements
    interpolator_x = cv2.resize(dx, (width, height), interpolation=cv2.INTER_CUBIC)
    interpolator_y = cv2.resize(dy, (width, height), interpolation=cv2.INTER_CUBIC)

    # Create sampling map.
    # Ensure that coordinates fit into the coordinate range of the input image.
    x_displaced = numpy.clip(x_fine + interpolator_x, 0, width - 1)
    y_displaced = numpy.clip(y_fine + interpolator_y, 0, height - 1)

    # Remap image and mask using the sampling maps.
    image = cv2.remap(
      image,
      x_displaced.astype(numpy.float32),
      y_displaced.astype(numpy.float32),
      interpolation=cv2.INTER_LINEAR,
      borderMode=cv2.BORDER_REFLECT
    )
    mask = cv2.remap(
      mask,
      x_displaced.astype(numpy.float32),
      y_displaced.astype(numpy.float32),
      interpolation=cv2.INTER_LINEAR,
      borderMode=cv2.BORDER_REFLECT
    )

    return Tensor(image).reshape(1, 1, width, height), Tensor(mask).reshape(1, 1, width, height)


if __name__ == "__main__":
  dataset = Dataset()
