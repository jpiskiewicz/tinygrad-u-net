#!/usr/bin/env python3

from PIL import Image
from tinygrad.tensor import Tensor
from tinygrad.nn.optim import SGD
from dataset import Dataset
from net import UNet
from error import pixel_error
from util import crop


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

if __name__ == "__main__":
  net = UNet()
  optimizer = SGD(net.weights, 0.01, 0.99)

  dataset = Dataset()

  with Tensor.train():
    for step in range(2000):
      batch, truth = dataset.choose()
      out = net(batch)
      if step % 10 == 0:
        Image.fromarray(out.numpy()[0][1] > 0).save(f"out/out_{step}.png")
      truth = crop(truth, out.shape[2])
      print("step:", step, "pixel error:", pixel_error(out, truth))
      loss = out.softmax().cross_entropy(truth)
      optimizer.zero_grad()
      loss.backward()
      optimizer.step()
