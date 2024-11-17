from tinygrad.tensor import Tensor
from tinygrad.ops import sint

def crop(t: Tensor, size: sint) -> Tensor:
  """
  Crops tensor to achieve the same size as another tensor.
  Used for stuff like contcatenating tenors or error
  calculations.
  """
  start = (t.shape[2] - size) // 2
  end = t.shape[2] - start
  assert isinstance(start, sint) and isinstance(end, sint)
  crop = (start, end)
  return t.shrink((None, None, crop, crop))
