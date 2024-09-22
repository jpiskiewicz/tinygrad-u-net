#!/usr/bin/env python3

from tinygrad.nn import Conv2d, BatchNorm, ConvTranspose2d

"""
An implementation of U-Net using tinygrad.
Based on https://arxiv.org/abs/1505.04597 and https://github.com/milesial/Pytorch-UNet.
Fueled by truckloads of Yerbata, way too many Serbian movies and ADHD meds.
"""

# Each convolutional layer of the U-Net consists of two conv blocks
# followed by a max pooling operation.
# Each conv block is made out of one 3x3 kernel convolution operation, batch norm and a ReLu.
class DoubleConv:
  def __init__(self, in_chan, out_chan):
    self.conv1 = Conv2d(in_chan, out_chan, 3, bias=False)
    self.bn = BatchNorm(out_chan)
    self.conv2 = Conv2d(out_chan, out_chan, 3, bias=False)

  def __call__(self, x):
    x = self.conv1(x)
    x = self.bn(x)
    x = x.relu()
    x = self.conv2(x)
    x = self.bn(x)
    return x.relu()

class EncoderLayer:
  def __init__(self, in_chan, out_chan):
    self.conv = DoubleConv(in_chan, out_chan)

  def __call__(self, x):
    return self.conv(x.max_pool2d())

class DecoderLayer:
  def __init__(self, in_chan, out_chan):
    self.transpose_conv = ConvTranspose2d(in_chan, in_chan, 2)
    self.conv = DoubleConv(in_chan, out_chan)

  def __call__(self, x):
    x = self.transpose_conv(x)
    return self.conv(x)

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

  def __call__(self, x):
    x = self.e1(x)
    x = self.e2(x)
    x = self.e3(x)
    x = self.e4(x)
    x = self.d1(x)
    x = self.d2(x)
    x = self.d3(x)
    x = self.d4(x)
    return self.final(x)
