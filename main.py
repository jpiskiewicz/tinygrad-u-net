#!/usr/bin/env python3

from tinygrad.nn import Conv2d, BatchNorm

"""
An implementation of U-Net using tinygrad.
Based on https://arxiv.org/abs/1505.04597 and https://github.com/milesial/Pytorch-UNet.
Fueled by truckloads of Yerbata, way too many Serbian movies and ADHD meds.
"""

# Each convolutional layer of the U-Net consists of two conv blocks
# followed by a max pooling operation.
# Each conv block is made out of one 3x3 kernel convolution operation, batch norm and a ReLu.
class DoubleConv:
  def __init__(self, in_channels, out_channels):
    self.conv1 = Conv2d(in_channels, out_channels, 3, bias=False)
    self.bn = BatchNorm(out_channels)
    self.conv2 = Conv2d(out_channels, out_channels, 3, bias=False)

  def __call__(self, x):
    x = self.conv1(x)
    x = self.bn(x)
    x = x.relu()
    x = self.conv2(x)
    x = self.bn(x)
    return x.relu()

class EncoderLayer:
  def __init__(self, in_channels, out_channels):
    self.conv = DoubleConv(in_channels, out_channels)

  def __call__(self, x):
    return self.conv(x).max_pool2d()

class UNet():
  def __init__(self):
    self.e1 = EncoderLayer(1, 64)
    self.e2 = EncoderLayer(64, 128)
    self.e3 = EncoderLayer(128, 256)
    self.e4 = EncoderLayer(256, 512)
    self.d1 = DecoderLayer(512, 1024)

  def __call__(self, x):
    x = self.e1(x)
    x = self.e2(x)
    x = self.e3(x)
    x = self.e4(x)
    x = self.d1(x)
