#!/usr/bin/env python3

"""
This is a place where I will develop smooth deformation algo written
in tinygrad tensor operations. It shouldn't be that hard, right?
it creates a dot grid and passes it first through the old
algo that utilizes OpenCV and then throught the new deformation
function written in tinygrad.
"""

import cv2
import numpy
from PIL import Image
from tinygrad.tensor import Tensor

from tinygrad import dtypes
from tinygrad_unet.dataset import SIZE
from tinygrad_unet.inference import mask_rgb

KERNEL_SIZE = 3
SD = 10


def make_dotted_image(width: int, height: int) -> Tensor:
    col_numbers = Tensor.arange(1, width + 1).repeat(height).reshape(width, height)
    padded = (col_numbers > width - 3).where(1, col_numbers)
    alternating_cols = padded.mod(4).sign().bitwise_xor(1)
    return alternating_cols.transpose().bitwise_and(alternating_cols)


def deform_opencv(image: Tensor, label: Tensor) -> tuple[Tensor, Tensor]:
    image = image.numpy().astype(numpy.float32)
    label = label.numpy().astype(numpy.float32)
    height, width = image.shape

    # Create displacement vectors
    dx, dy = [numpy.random.normal(0, SD, (KERNEL_SIZE, KERNEL_SIZE)) for _ in range(2)]

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
        borderMode=cv2.BORDER_REFLECT,
    )
    label = cv2.remap(
        label,
        x_displaced.astype(numpy.float32),
        y_displaced.astype(numpy.float32),
        interpolation=cv2.INTER_NEAREST,
        borderMode=cv2.BORDER_REFLECT,
    )

    return Tensor(image).reshape(1, 1, width, height), Tensor(label).reshape(
        1, 1, width, height
    )


def source_coords(in_size: int, out_size: int) -> Tensor:
    dst = Tensor.arange(out_size, dtype=dtypes.float)


def kernel_upscale(kernel: Tensor, dimensions: Tuple[int, int]) -> Tensor:
    """
    Perform bicubic interpolation on the kernel in order to bring it up to the provided dimensions.
    """


def nearest_neighbour_interpolation(image: Tensor, xs: Tensor, ys: Tensor) -> Tensor:
    return image[ys.round(), xs.round()]


def deform_tiny(image: Tensor, label: Tensor) -> Tensor:
    kernel = Tensor.normal((KERNEL_SIZE, KERNEL_SIZE), mean=0, std=SD)


def tensor_to_image(t: Tensor) -> Image.Image:
    return Image.fromarray(mask_rgb(t, (255, 255, 0))).convert("RGB")


if __name__ == "__main__":
    orig = make_dotted_image(SIZE, SIZE)
    tensor_to_image(orig.reshape(1, 1, orig.shape[0], orig.shape[1])).save(
        "deform_before.png"
    )
    image_opencv, label_opencv = deform_opencv(orig, orig)
    tensor_to_image(image_opencv).save("deform_opencv_image.png")
    tensor_to_image(label_opencv).save("deform_opencv_label.png")
    deform_tiny(orig)
