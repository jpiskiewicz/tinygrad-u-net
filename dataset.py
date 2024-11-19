#!/usr/bin/env python3

import numpy
import cv2
import os
import glob
from tinygrad.tensor import Tensor
from PIL import Image
from multiprocessing import Pool
from tqdm import tqdm
from typing import Union


class Dataset:
  def __init__(self):
    dirs = ["benign", "malignant", "normal"]
    files = self.get_files(dirs)
    images, masks = map(lambda x: self.read_image(x), files[0]), map(lambda x: self.read_image(x, "1"), files[1])
    masks = self.combine_masks(masks)

  def collect_glob(self, dirs: list[str], is_mask: bool = False) -> list[str]:
    files = []
    for x in dirs: files += glob.glob(f"data/{x}/*{'' if is_mask else '[!_]'}).png")
    return files

  def get_files(self, dirs: list[str]) -> tuple[list[str], list[str]]:
    return self.collect_glob(dirs), self.collect_glob(dirs, True)

  def read_image(self, filename: str, mode: str = "L") -> numpy.ndarray:
    image = Image.open(filename).convert(mode)
    center = [x/2 for x in image.size]
    cropped = numpy.array(image.crop((center[0] - 286, center[1] - 286, center[0] + 286, center[1] + 286))).astype(numpy.uint8)
    return cropped

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
