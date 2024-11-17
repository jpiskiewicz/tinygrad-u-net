#!/usr/bin/env python3

import numpy
import sys
import cv2
import os
import re
import glob
from tinygrad.tensor import Tensor
from PIL import Image
from multiprocessing import Pool
from tqdm import tqdm
from typing import Union
from random import shuffle


type ImageWithGroundTruth = tuple[Tensor, Tensor]


class Dataset:
  def __init__(self):
    dirs = ["benign", "malignant", "normal"]
    files = []
    for dir in dirs:
      current_dir = os.path.join("data", dir)
      batch = [os.path.join(current_dir, x) for x in os.listdir(current_dir)]
      files = [*files, *filter(lambda x: re.search("_mask*", x) is None, batch)]
    data = self.preprocess(files)
    shuffle(data)
    train_size = int(len(data) * 0.6)
    self.train, self.val = data[:train_size], data[train_size:]

  def convert_and_crop(self, image: Image.Image, mode: str = "L", feature_id: Union[int, None] = None) -> numpy.ndarray:
    image = image.convert(mode)
    center = [x/2 for x in image.size]
    cropped = numpy.array(image.crop((center[0] - 286, center[1] - 286, center[0] + 286, center[1] + 286))).astype(numpy.uint8)
    if mode == "1":
      assert feature_id is not None
      cropped[cropped > 0] += feature_id
    return cropped

  def deform(self, image: numpy.ndarray, mask: numpy.ndarray) -> tuple[Tensor, Tensor]:
    """
    This contains the logic for smooth image deformation (https://en.wikipedia.org/wiki/Homotopy)
    using random displacement vectors on a coarse 3x3 grid (see section 3.1 in the U-Net paper).
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

  def get_mask(self, path: str) -> numpy.ndarray:
    root, ext = os.path.splitext(path)
    filenames = glob.glob(root + "_mask*")
    read = lambda filename, i: self.convert_and_crop(Image.open(filename), "1", i)
    mask = read(filenames[0], 0)
    for i, filename in enumerate(filenames[1:]):
      mask += read(filename, i + 1)
    return mask

  def split_mask(self, mask: Tensor) -> Tensor:
    """
    Split feature mask which contains object IDs into two channels:
    binary channel and an integer-valued object ID channel.
    """
    return mask.cat(mask.clamp(Tensor.full(mask.shape, 0), Tensor.full(mask.shape, 1)), dim=1)

  def preprocess_internal(self, path: str) -> ImageWithGroundTruth:
    image = Image.open(path)
    return self.deform(self.convert_and_crop(image), self.get_mask(path))

  def preprocess(self, files: list[str]) -> list[ImageWithGroundTruth]:
    progress = tqdm(total=len(files), desc="Loading dataset")
    data = []
    with Pool(8) as pool:
      async_result = pool.map(self.preprocess_internal, files)
      for res in async_result:
        progress.update()
        data.append(res)
      progress.close()
      return data

  def choose(self) -> ImageWithGroundTruth:
    i = numpy.random.randint(0, len(self.train))
    batch, truth = self.train[i]
    truth = self.split_mask(truth)
    return batch, truth
