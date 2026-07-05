#!/usr/bin/env python3

"""
This is a place where I will develop smooth deformation algo written
in tinygrad tensor operations. It shouldn't be that hard, right?
"""

from tinygrad.tensor import Tensor
from tinygrad_unet.dataset import SIZE


def make_dotted_image(width: int, height: int) -> Tensor:
  return Tensor.arange(1, width*height + 1).reshape(width, height).mod(4).sign().bitwise_xor(1)
  
if __name__ == "__main__":
  print(make_dotted_image(SIZE, SIZE).numpy())