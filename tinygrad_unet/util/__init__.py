from tinygrad.tensor import Tensor
from tinygrad.dtype import dtypes
import numpy

def make_8bit(im: Tensor) -> numpy.ndarray: return (im * 255).cast(dtypes.uint8)[0][0].numpy()
