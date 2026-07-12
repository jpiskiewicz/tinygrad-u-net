#!/usr/bin/env python3

"""
This is a place where I will develop smooth deformation algo written
in tinygrad tensor operations. It shouldn't be that hard, right?
it creates a dot grid and passes it first through the old
algo that utilizes OpenCV and then throught the new deformation
function written in tinygrad.
"""

from tinygrad.tensor import Tensor
from tinygrad_unet.dataset import SIZE
from tinygrad_unet.inference import mask_rgb
import numpy
import cv2
from PIL import Image


def make_dotted_image(width: int, height: int) -> Tensor:
  col_numbers = Tensor.arange(1, width + 1).repeat(height).reshape(width, height)
  padded = (col_numbers > width - 3).where(1, col_numbers)
  alternating_cols = padded.mod(4).sign().bitwise_xor(1)
  return alternating_cols.transpose().bitwise_and(alternating_cols)
  
  
def deform_opencv(image: Tensor) -> Tensor:
    image = image.numpy().astype(numpy.float32)
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

    return Tensor(image).reshape(1, 1, width, height)
 
# def deform_tiny(image: Tensor) -> Tensor:


def tensor_to_image(t: Tensor) -> Image.Image:
  return Image.fromarray(mask_rgb(t, (255, 255, 0))).convert("RGB")


if __name__ == "__main__":
  orig = make_dotted_image(SIZE, SIZE)
  print(orig.numpy())
  tensor_to_image(orig.reshape(1, 1, orig.shape[0], orig.shape[1])).save("deform_before.png")
  tensor_to_image(deform_opencv(orig)).save("deform_opencv.png")