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


def pixel_error(out: Tensor, truth: Tensor) -> float:
  """
  Pixel error is just the Euclidean distance between two binary-valued
  tensors.
  """
  return (out[0][0] - truth[0][0]).square().mean().numpy().item()
