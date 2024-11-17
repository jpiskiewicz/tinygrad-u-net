from tinygrad.nn import Conv2d, BatchNorm, ConvTranspose2d
from tinygrad.nn.state import safe_save, get_state_dict
from tinygrad.tensor import Tensor
from tinygrad.ops import sint
from util import crop


class DoubleConv:
  """
  Each convolutional layer of the U-Net consists of two conv blocks
  followed by a max pooling operation.
  Each conv block is made out of one 3x3 kernel convolution operation, batch norm and a ReLu.
  """

  def __init__(self, in_chan: int, out_chan: int):
    self.conv1 = Conv2d(in_chan, out_chan, 3, bias=False)
    self.bn = BatchNorm(out_chan)
    self.conv2 = Conv2d(out_chan, out_chan, 3, bias=False)

  def __call__(self, x: Tensor) -> Tensor:
    x = self.conv1(x)
    x = self.bn(x)
    x = x.relu()
    x = self.conv2(x)
    x = self.bn(x)
    return x.relu()

  def weights(self) -> list[Tensor]:
    return [self.conv1.weight, self.conv2.weight]


class EncoderLayer:
  def __init__(self, in_chan, out_chan):
    self.conv = DoubleConv(in_chan, out_chan)

  def __call__(self, x: Tensor) -> Tensor:
    return self.conv(x.max_pool2d(stride = 2))

  def weights(self) -> list[Tensor]:
    return self.conv.weights()


class DecoderLayer:
  def __init__(self, in_chan, out_chan):
    self.transpose_conv = ConvTranspose2d(in_chan, out_chan, 2, stride = 2)
    self.conv = DoubleConv(in_chan, out_chan) # in_chan because we do concat with contracting layer

  def __call__(self, x: Tensor, c: Tensor) -> Tensor:
    """
    c is the output from one of the layers of the contracting path.
    We need to crop it before we concatenate it with the current level
    from expanding path.
    """
    x = self.transpose_conv(x)
    c = crop(c, x.shape[2])
    x = x.cat(c, dim = 1)
    return self.conv(x)

  def weights(self) -> list[Tensor]:
   return [self.transpose_conv.weight, *self.conv.weights()]

class UNet():
  def __init__(self):
    self.initial = DoubleConv(1, 64)
    self.e1 = EncoderLayer(64, 128)
    self.e2 = EncoderLayer(128, 256)
    self.e3 = EncoderLayer(256, 512)
    self.e4 = EncoderLayer(512, 1024)
    self.d1 = DecoderLayer(1024, 512)
    self.d2 = DecoderLayer(512, 256)
    self.d3 = DecoderLayer(256, 128)
    self.d4 = DecoderLayer(128, 64)
    self.final = Conv2d(64, 2, 1)

  def __call__(self, x) -> Tensor:
    x1 = self.initial(x)
    x2 = self.e1(x1)
    x3 = self.e2(x2)
    x4 = self.e3(x3)
    x = self.e4(x4)
    x = self.d1(x, x4)
    x = self.d2(x, x3)
    x = self.d3(x, x2)
    x = self.d4(x, x1)
    return self.final(x)

  def save_state(self):
    safe_save(get_state_dict(self), "checkpoint.safetensor")

  @property
  def weights(self) -> list[Tensor]:
   return self.initial.weights()
   + self.e1.weights() + self.e2.weights() + self.e3.weights() + self.e4.weights()
   + self.d1.weights() + self.d2.weights() + self.d3.weights() + self.d4.weights()
   + self.final.weights()
