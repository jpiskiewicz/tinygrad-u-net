#!/usr/bin/env python3

from PIL import Image
from tinygrad import TinyJit
from tinygrad.tensor import Tensor
from tinygrad.nn.optim import SGD
from dataset import Dataset, save_image
from net import UNet
from error import pixel_error
from util import crop
from typing import Callable


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


def get_test_predictor(net: UNet, dataset: Dataset) -> Callable[[int, Tensor], None]:
  image, truth = dataset.images[0], dataset.masks[0]
  save_image(image, "out/batch.png")
  save_image(truth, "out/truth.png", mask = True)
  def f(step: int, loss: Tensor):
    Tensor.training = False
    out = net(image)
    print("step:", step, "loss:", loss.numpy(), "pixel error:", pixel_error(out, crop(truth, out.shape[2])))
    Image.fromarray(out.numpy()[0][0] > 0).save(f"out/out_{step}.png")
  return f


if __name__ == "__main__":
  net = UNet()
  optimizer = SGD(net.weights, 0.01, 0.99)
  dataset = Dataset()

  save_test_prediction = get_test_predictor(net, dataset)

  @TinyJit
  def perform_train_step():
    Tensor.training = True
    samp = Tensor.randint(1)
    batch, truth = dataset.images[samp[0].numpy()], dataset.masks[samp[0].numpy()]
    out = net(batch)
    truth = crop(truth, out.shape[2])
    loss = out.softmax().cross_entropy(truth)
    optimizer.zero_grad()
    loss = loss.backward()
    optimizer.step()
    return loss

  for step in range(100000):
    loss = perform_train_step()
    if step % 100 == 0:
      save_test_prediction(step, loss)
    if step % 1000 == 0:
      net.save_state()
