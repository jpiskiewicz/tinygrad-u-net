#!/usr/bin/env python3

import sys
import os
import numpy
import re
from PIL import Image
from random import shuffle
from tinygrad.tensor import Tensor
from tinygrad.nn.optim import SGD
from input_transform import preprocess, ImageWithGroundTruth
from net import UNet
from error import pixel_error


"""
An implementation of U-Net using tinygrad.
Based on https://arxiv.org/abs/1505.04597 and https://github.com/milesial/Pytorch-UNet.
Fueled by truckloads of Yerbata, way too many Serbian movies and ADHD meds.
"""

# TODO)) Investigate whether adding the dice score to the loss
# (like in the Pytorch-UNet implementation) helps with training.
# TODO)) Find out what kind of weight map generation scheme would be
# effective on ultrasound images.
# TODO)) Warping error, Rand Error, Pixel Error.
# DONE: Understand what kind of deformations are done to the data in the whitepaper.

def get_data() -> tuple[list[ImageWithGroundTruth], list[ImageWithGroundTruth]]:
  dirs = ["benign", "malignant", "normal"]
  files = []
  for dir in dirs:
    current_dir = os.path.join("data", dir)
    batch = [os.path.join(current_dir, x) for x in os.listdir(current_dir)]
    files = [*files, *filter(lambda x: re.search("_mask*", x) is None, batch)]
  data = preprocess(files)
  shuffle(data)
  train_size = int(len(data) * 0.6)
  train, val = data[:train_size], data[train_size:]
  return train, val


if __name__ == "__main__":
  net = UNet()
  optimizer = SGD(net.weights, 0.01, 0.99)

  train, val = get_data()

  with Tensor.train():
    for step in range(2000):
      i = numpy.random.randint(0, len(train))
      batch, truth = train[i]
      out = net(batch)
      print(out.numpy())
      print("step:", step, "pixel error:", pixel_error(out, truth))
      loss = out.softmax().cross_entropy(truth)
      optimizer.zero_grad()
      optimizer.step()
