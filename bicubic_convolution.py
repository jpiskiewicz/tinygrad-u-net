#!/usr/bin/env python3

"""
The purpose of this file is to understand how bicubic convolution works.
Initialy I will create a sequential 2D algorithm and then I'll rewrite
it so that it uses a convolution operation on tensors.
"""

from PIL import Image
import numpy
import math

def kernel(s: float) -> float:
  if 0 < s and s < 1: return 3 / 2 * abs(s)**3 - 5 / 2 * abs(s)**2 + 1
  elif 1 < s and s < 2: return - 1 / 2 * abs(s)**3 + 5 / 2 * abs(s)**2 - 4 * abs(s) + 2
  return 0

def translate_coord(coord: int, scale: float) -> int: return int(coord / scale)

def bound(src: numpy.ndarray, x: int, y: int) -> int:
  if x == -1:
    return 3 * bound(src, 0, y) - 3 * bound(src, 1, y) + bound(src, 2, y)
  elif x == src.shape[0]:
    return 3 * bound(src, src.shape[0] - 1, y) - 3 * bound(src, src.shape[0] - 2, y) + bound(src, src.shape[1] - 3, y)
  elif y == -1:
    return 3 * bound(src, x, 0) - 3 * bound(src, x, 1) + bound(src, x, 2)
  elif y == src.shape[1]:
    return 3 * bound(src, x, src.shape[1] - 1) - 3 * bound(src, x, src.shape[1] - 2) + bound(src, x, src.shape[1] - 3)
  return src[x, y]

def out_pixel(x: int, y: int, src: numpy.ndarray, scale: float) -> int:
  x_src, y_src = translate_coord(x, scale), translate_coord(y, scale)
  out = 0
  for l in range(-1, src.shape[0] + 1):
    for m in range(-1, src.shape[1] + 1):
      out += int(bound(src, l, m) * kernel((x_src - l) / scale) * kernel((y_src - m) / scale))
  return out

def upscale(src: Image.Image, size: int) -> Image.Image:
  source = numpy.array(src)
  scale = size / source.shape[0]
  out = numpy.ndarray((size, size), numpy.uint8)
  for x in range(size):
    for y in range(size):
      out[x, y] = out_pixel(x, y, source, scale)
      percent = (x + 1) * (y + 1) / size**2 * 100
      if percent % 1 == 0: print(x, y, percent)
  return Image.fromarray(out)

if __name__ == "__main__":
  sine_image = numpy.ndarray((100, 100), numpy.uint8)
  for i in range(sine_image.shape[0]):
    for j in range(sine_image.shape[1]):
      print(i, j, math.sqrt(i**2 + j**2))
      sine_image[i, j] = math.sin(math.sqrt(0.5 * ((i/100)**2 + (j/100)**2)**2)) * 255 # TODO)) This function is to smooth to be used for this purpose
  src = Image.fromarray(sine_image)
  src.save("src.jpeg")
  out = upscale(src, 200)
  out.save("upscaled.jpeg")
