"""
Functions for computing different types of error rates for the segmentation problem.
"""

from tinygrad.tensor import Tensor


def warping_error(t1: Tensor, t2: Tensor) -> float:
  """
  Reference:
  https://github.com/fiji/Trainable_Segmentation/blob/41c7083382816b8d40deffbb06c9c3eace0b1e55/src/main/java/trainableSegmentation/metrics/WarpingError.java
  """
  return -1


def  rand_error(t1: Tensor, t2: Tensor) -> float:
  return -1


def pixel_error(t1: Tensor, t2: Tensor) -> float:
  """
  Pixel error is just the Euclidean distance between two binary-valued
  tensors.
  """
  print(t1.shape, t2.shape)
  return (t1 - t2).square().mean().numpy().item()
