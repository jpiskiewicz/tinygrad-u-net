from tinygrad.nn import Conv2d, BatchNorm, ConvTranspose2d
from tinygrad.nn.state import safe_save, get_state_dict
from tinygrad.tensor import Tensor


class DoubleConv:
  """
  Each convolutional layer of the U-Net consists of two conv blocks
  followed by a max pooling operation.
  Each conv block is made out of one 3x3 kernel convolution operation, batch norm and a ReLu.
  """

  def __init__(self, in_chan: int, out_chan: int, upsample: bool = False):
    self.conv1 = Conv2d(in_chan, out_chan, 3, stride=1 if upsample else 2, padding=1, bias=False)
    self.bn = BatchNorm(out_chan)
    self.conv2 = Conv2d(out_chan, out_chan, 3, padding=1, bias=False)

  def __call__(self, x: Tensor) -> Tensor:
    x = self.conv1(x)
    x = self.bn(x)
    x = x.relu()
    x = self.conv2(x)
    x = self.bn(x)
    return x.relu()


class DecoderLayer:
  def __init__(self, in_chan, out_chan):
    self.transpose_conv = ConvTranspose2d(in_chan, out_chan, 2, stride=2)
    self.conv = DoubleConv(in_chan, out_chan, True) # in_chan because we do concat with contracting layer

  def __call__(self, x: Tensor, c: Tensor) -> Tensor:
    x = self.transpose_conv(x)
    x = x.cat(c, dim = 1)
    return self.conv(x)
    

class UNet():
  def __init__(self):
    self.initial = DoubleConv(1, 64, True)
    self.e1 = DoubleConv(64, 128)
    self.e2 = DoubleConv(128, 256)
    self.e3 = DoubleConv(256, 512)
    self.e4 = DoubleConv(512, 1024)
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
