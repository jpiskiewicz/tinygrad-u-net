#!/usr/bin/env python3

from os import path
from sys import argv
from PIL import Image
from tinygrad.tensor import Tensor
from tinygrad.nn.state import load_state_dict, safe_load
from tinygrad import dtypes 
from dataset import load_image, transform_image, get_masks, load_mask
from net import UNet
from pathlib import Path
import json


def load_and_transform(filename: str): return transform_image(load_image(filename))

def load_combined_mask(filename: str): return load_mask([load_image(x) for x in get_masks(filename)])

def make_8bit(im: Tensor) -> Tensor: return ((im - im.min()) / (im.max() - im.min()) * 255).cast(dtypes.uint8)[0][0].numpy()

def mask_rgb(mask: Tensor, color: tuple[int, int, int]) -> Tensor: return (mask.unsqueeze(4).repeat_interleave(4, 4)[0][0] * Tensor(color + (255,))).cast(dtypes.uint8).numpy()


def infer_and_overlap(net: UNet, filename: str, subdir: str, epoch: int = 0):
    im = load_and_transform(filename + ".png")
    print("Generating prediction...")
    pred = net(im)
    display_pred = Image.fromarray(mask_rgb(pred.sigmoid() > 0.5, (0, 0, 255)))
    display_im = Image.fromarray(make_8bit(im)).convert("RGB")
    original_mask = Image.fromarray(mask_rgb(load_combined_mask(filename), (255, 255, 0)))
    out_path = Path(path.join("predictions", subdir, f"{path.split(filename)[1]}_{epoch}.png"))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    Image.blend(display_im, Image.alpha_composite(original_mask, display_pred).convert("RGB"), 0.5).save(out_path)
    print("Saved prediction at", out_path)
  

if __name__ == "__main__":
    if len(argv) != 4:
        print('Please provide a model checkpoint file in .safetensors format, mode ("filelist", "dirname") and the directory containing the .nii file.')
        exit(1)
    net = UNet()
    state = safe_load(argv[1])  # Load model checkpoint
    load_state_dict(net, state)
    if argv[2] == "filelist": 
        with open(argv[3]) as f: dirnames = json.load(f)
    else: dirnames = [argv[3]]
    for dirname in dirnames: infer_and_overlap(net, dirname, "inference")
