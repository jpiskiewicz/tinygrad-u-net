#!/usr/bin/env python3

import numpy
import sys
import cv2
import os
import glob
from PIL import Image


"""
This contains the logic for smooth image deformation (https://en.wikipedia.org/wiki/Homotopy)
using random displacement vectors on a coarse 3x3 grid (see section 3.1 in the U-Net paper).
"""

def convert_and_crop(image: Image.Image) -> numpy.ndarray:
  image = image.convert("L")
  center = [x/2 for x in image.size]
  return numpy.array(image.crop((center[0] - 240, center[1] - 240, center[0] + 240, center[1] + 240)))


def deform(image: numpy.ndarray, mask: numpy.ndarray) -> tuple[numpy.ndarray, numpy.ndarray]:
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

  return image, mask


def get_mask(path: str) -> numpy.ndarray:
  root, ext = os.path.splitext(path)
  filenames = glob.glob(root + "_mask*")
  print("get_mask; filenames:", filenames)
  read = lambda filename: convert_and_crop(Image.open(filename))
  mask = read(filenames[0])
  for filename in filenames[1:]:
    mask += read(filename)
  return mask


def preprocess(path: str) -> tuple[numpy.ndarray, numpy.ndarray]:
  image = Image.open(path)
  return deform(convert_and_crop(image), get_mask(path))


if __name__ == "__main__":
  """
  This module can be invoked from a command line with path to image file.
  It then outputs an edited image at specified output path.
  This has no use other than testing.
  """
  input, output = sys.argv[1:3]
  processed = [Image.fromarray(x) for x in preprocess(input)]
  image, mask = processed
  image.save(output + ".png")
  mask.save(output + "_mask.png")
