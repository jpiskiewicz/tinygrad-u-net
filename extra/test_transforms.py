#!/usr/bin/env python3

from tinygrad_unet.dataset import TrivialAugument, TensorTransform, load_dataset
from tinygrad_unet.inference import mask_rgb
from tinygrad_unet.util import make_8bit
from tinygrad.tensor import Tensor
from PIL import Image, ImageDraw, ImageFont
from typing import override
from pathlib import Path
import argparse


PADDING_TOP = 32
GRID_GAP = 4
COLS = 4
FONT = ImageFont.truetype(Path(__file__).resolve().parent / "fonts/noto.ttf", size=16)


def arrange_on_grid(images: list[Image.Image], titles: list[str], output: str):
    image_width, image_height = images[0].size
    rows = len(images) // COLS
    cell_height = image_height + GRID_GAP + PADDING_TOP
    grid_width, grid_height = image_width * COLS + (GRID_GAP * (COLS + 1)), (image_height + GRID_GAP + PADDING_TOP) * rows
    res = Image.new("RGB", (grid_width, grid_height), color="white")
    for i in range(COLS):
        for j in range(rows):
            ImageDraw.Draw(res).text((image_width * i + (GRID_GAP * (i + 1)) + image_width // 2, cell_height * j), titles[i*rows+j], (0, 0, 0), FONT, "ma")
            res.paste(images[i*rows+j], (image_width * i + (GRID_GAP * (i + 1)), cell_height * j + PADDING_TOP))
    res.save(output)
    
  
class Original(TensorTransform):
   @override
   def apply(self, image: Tensor) -> Tensor: return image
  
  
def convert_to_image(tensors: tuple[Tensor, Tensor]) -> tuple[Image.Image, Image.Image]:
  return Image.fromarray(make_8bit(tensors[0])).convert("RGB"), Image.fromarray(mask_rgb(tensors[1], (255, 255, 0)))

  
if __name__ == "__main__":
  parser = argparse.ArgumentParser(prog="test_transforms", description="Runs all transformations from TrivialAugument on a image-mask pair at index = index.")
  parser.add_argument('dataset')
  parser.add_argument('index', type=int)
  parser.add_argument('output')
  parser.add_argument('-m', '--magnitude', type=float, default=0.5)
  args = parser.parse_args()
  
  print("Loading dataset...")
  dataset = load_dataset(args.dataset)
  print("Creating TrivialAugument and converting dataset to Pillow Image instances...")
  aug = TrivialAugument([[dataset[0][args.index]], [dataset[1][args.index]]])
  print("Running transforms...")
  transform_builders = [Original, *aug.transformations]
  transforms = [x(args.magnitude) for x in transform_builders]
  images = [x for pair in [convert_to_image(aug.run_transform(0, transform)) for transform in transforms] for x in pair]
  
  print("Drawing the grid...")
  titles = [x for pair in [[x.__name__ + " image", x.__name__ + " mask"] for x in transform_builders] for x in pair]
  arrange_on_grid(images, titles, args.output)
  print("Done!")