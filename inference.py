#!/usr/bin/env python3

from os import path
from sys import argv
from PIL import Image
from tinygrad.tensor import Tensor
from tinygrad.nn.state import load_state_dict, safe_load
from tinygrad import dtypes 
from dataset import load_slice
from net import UNet
from pathlib import Path
import json

def make_8bit(im: Tensor) -> Tensor: return ((im - im.min()) / (im.max() - im.min()) * 255).cast(dtypes.uint8)[0][0].numpy()

def mask_rgb(mask: Tensor, color: tuple[int, int, int]) -> Tensor: return (mask.unsqueeze(4).repeat_interleave(4, 4)[0][0] * Tensor(color + (255,))).cast(dtypes.uint8).numpy()

def infer_and_overlap(net: UNet, dirname: str, subdir: str):
  im = load_slice(path.join(dirname, Path(dirname).name + "_t1ce.nii"), False)
  print("Generating prediction...")
  pred = net(im)
  display_pred = Image.fromarray(mask_rgb(pred.sigmoid() > 0.5, (0, 0, 255)))
  display_im = Image.fromarray(make_8bit(im)).convert("RGB")
  original_mask = Image.fromarray(mask_rgb(load_slice(path.join(dirname, Path(dirname).name + "_seg.nii"), True), (255, 255, 0)))
  out_path = Path(path.join("predictions", subdir, path.basename(dirname) + ".png"))
  out_path.parent.mkdir(parents=True, exist_ok=True)
  Image.blend(display_im, Image.alpha_composite(display_pred, original_mask).convert("RGB"), 0.5).save(out_path)
  print("Saved prediction.")
  

if __name__ == "__main__":
    # if len(argv) != 3:
    #     print("Please provide a model checkpoint file in .safetensors format and the .nii file.")
    #     exit(1)
    # net = UNet()
    # state = safe_load(argv[1])  # Load model checkpoint
    # load_state_dict(net, state)
    # with open(argv[2]) as f: dirnames = json.load(f)
    # for dirname in dirnames: infer_and_overlap(net, dirname, path.splitext(argv[2])[0])
