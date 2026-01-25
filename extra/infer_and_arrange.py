#!/usr/bin/env python3

import json
import numpy
from sys import argv
from tinygrad.dtype import dtypes
from tinygrad.tensor import Tensor
from tinygrad.nn.state import load_state_dict, safe_load
from PIL import Image, ImageDraw, ImageFont
from tinygrad_unet.inference import run_inference
from tinygrad_unet.net import UNet

"""
This tool takes a list of images in a JSON format, runs inference using
the chosen model on them and draws a grid of predictions. It supports
naming the grid (via command-line arguments) so that the user can
specify the model which created the predictions for easier reference.
"""


PADDING_TOP = 32
GRID_GAP = 4
COLS = 2


# def prediction_probability_gradient(pred: Tensor) -> numpy.ndarray:
#     p = pred[0][0]
#     half = Tensor(0.5)
#     t1 = p / half
#     t2 = (p - half) / half
#     r = t2.clip(0, 1) * 255
#     g = Tensor.where(p <= half, t1 * 255, (1 - t2).clip(0, 1) * 255)
#     b = Tensor.where(p <= half, (1 - t1).clip(0, 1) * 255, Tensor(0))
#     alpha = p * 255
#     rgba = Tensor.stack(r, g, b, alpha, dim=2)
#     return rgba.cast(dtypes.uint8).numpy()


if __name__ == "__main__":
    net = UNet()
    load_state_dict(net, safe_load(argv[1]))
    with open(argv[2]) as f: dirnames = json.load(f)
    images = [run_inference(net, "../" + x) for x in dirnames]
    image_width, image_height = images[0].size
    rows = len(images) // COLS
    grid_width, grid_height = image_width * COLS + (GRID_GAP * (COLS + 1)), (image_height + GRID_GAP) * rows
    res = Image.new("RGB", (grid_width, grid_height + PADDING_TOP), color="white")
    for i in range(COLS):
        for j in range(rows):
            res.paste(images[i*rows+j], (image_width * i + (GRID_GAP * (i + 1)), PADDING_TOP + (image_height + GRID_GAP) * j))
    font = ImageFont.truetype("fonts/noto.ttf", size=16)
    ImageDraw.Draw(res).text((grid_width / 2, PADDING_TOP / 2), argv[3], (0, 0, 0), font, "mm")
    res.save(argv[4])

