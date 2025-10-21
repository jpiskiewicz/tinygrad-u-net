from tinygrad.tensor import Tensor
from matplotlib import pyplot as plt

def plot_slice(slice: Tensor):
  _, ax = plt.subplots(figsize=(16, 3))
  ax.imshow(slice.numpy())
  plt.show()