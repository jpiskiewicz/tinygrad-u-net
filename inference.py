#!/usr/bin/env python3

from net import UNet
from sys import argv
from tinygrad.nn.state import safe_load, load_state_dict
from dataset import read_image
from PIL import Image
from os import path

if __name__ == "__main__":
  if len(argv) != 2:
    print("Please provide a model checkpoint file in .safetensors format as an argument.")
    exit(1)
  net = UNet()
  state = safe_load(argv[1]) # Load model checkpoint
  load_state_dict(net, state)
  while True:
    name, im = read_image((input("Please provide image to make prediction out of: "), False))
    print("Generating prediction...")
    out = net(im)
    Image.fromarray(out.sigmoid()[0][0].numpy() > 0.5).save(path.splitext(path.basename(name))[0] + ".png")
    print("Saved prediction.")