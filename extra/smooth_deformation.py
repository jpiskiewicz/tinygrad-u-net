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
from tinygrad.uop.ops import sint

from tinygrad import dtypes
from tinygrad_unet.inference import mask_rgb

KERNEL_SIZE = 3
SD = 10
SIZE = 240


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
    """
    This is a sampling map which pairs the output pixels to source pixels.
    The values can be fractional which means that a given position in the output
    tensor maps to some place in between two pixels in the source tensor.
    """
    return Tensor.arange(out_size, dtype=dtypes.float).add(0.5).div(out_size/in_size).sub(0.5)


def cubic_close(x: Tensor, a: float) -> Tensor: return ((a + 2) * x - (a + 3)) * x**2 + 1


def cubic_far(x: Tensor, a: float) -> Tensor: return ((a * x - 5 * a) * x + 8 * a) * x - 4 * a


def cubic_weights(t: Tensor, a: float = -0.5) -> tuple[Tensor, ...]:
    return cubic_far(t + 1, a), cubic_close(t, a), cubic_close(1 - t, a), cubic_far(2 - t, a)


def kernel_upscale(kernel: Tensor, dimensions: tuple[sint, ...]) -> Tensor:
    """
    Perform bicubic interpolation on the kernel in order to bring it up to the provided dimensions.
    """
    x = kernel
    for i in range(2): # Bicubic convolution is a separable operation
        src = source_coords(int(kernel.shape[i]), int(dimensions[i]))
        base = src.floor().cast(dtypes.int)
        t = src - base
        weights = cubic_weights(t)
        weight_dims = dimensions[0] if i == 0 else 1, dimensions[1] if i == 1 else 1, 1
        index_dims = dimensions[0], dimensions[1] if i == 1 else kernel.shape[1], 2
        out = Tensor.zeros(index_dims)
        for offset, weight in zip((-1, 0, 1, 2), weights):
            idx = (base + offset).clip(0, x.shape[i] - 1).reshape(weight_dims).expand(index_dims)
            samples = x.gather(-3 + i, idx)
            out += samples * weight.reshape(weight_dims).expand(index_dims)
        x = out
    return x


def nearest_neighbour_interpolation(image: Tensor, xs: Tensor, ys: Tensor) -> Tensor:
    return image[ys.round().cast(dtypes.int), xs.round().cast(dtypes.int)].reshape(1, 1, *image.shape)


def deform_tiny(image: Tensor, label: Tensor) -> Tensor:
    kernel = Tensor.normal((KERNEL_SIZE, KERNEL_SIZE, 2), mean=0, std=SD)
    shift_map = kernel_upscale(kernel, image.shape)
    # TODO)) Think about whether it would be better (for the NB remapping here and bicubic remapping later on)
    # to move the last dimension of the coordinate shift map to the front.
    shift_map_x = shift_map.gather(2, Tensor(0).repeat(image.shape[0] * image.shape[1]).reshape(*image.shape, 1)).reshape(image.shape)
    shift_map_y = shift_map.gather(2, Tensor(1).repeat(image.shape[0] * image.shape[1]).reshape(*image.shape, 1)).reshape(image.shape)
    xs = Tensor.arange(image.shape[1]).reshape(1, image.shape[1]).expand(image.shape).add(shift_map_x).clip(0, image.shape[1] - 1)
    ys = Tensor.arange(image.shape[0]).reshape(image.shape[0], 1).expand(image.shape).add(shift_map_y).clip(0, image.shape[1] - 1)
    return nearest_neighbour_interpolation(image, xs, ys)


def tensor_to_image(t: Tensor) -> Image.Image:
    return Image.fromarray(mask_rgb(t, (255, 255, 0))).convert("RGB")


if __name__ == "__main__":
    orig = make_dotted_image(SIZE, SIZE)
    tensor_to_image(orig.reshape(1, 1, orig.shape[0], orig.shape[1])).save(
        "deform_before.png"
    )
    image_opencv, label_opencv = deform_opencv(orig, orig)
    print(label_opencv.shape)
    tensor_to_image(image_opencv).save("deform_opencv_image.png")
    tensor_to_image(label_opencv).save("deform_opencv_label.png")
    label_tiny = deform_tiny(orig, orig)
    tensor_to_image(label_tiny).save("deform_tiny_label.png")
