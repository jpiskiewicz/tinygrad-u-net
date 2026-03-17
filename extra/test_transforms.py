#!/usr/bin/env python3

from tinygrad_unet.dataset import TrivialAugument, Transform, load_dataset
from tinygrad_unet.inference import mask_rgb
from tinygrad_unet.util import make_8bit
from PIL import Image, ImageDraw, ImageFont
import argparse


PADDING_TOP = 32
GRID_GAP = 4
COLS = 4
FONT = ImageFont.truetype("fonts/noto.ttf", size=16)


def run_transform(aug: TrivialAugument, transform: Transform, image: Image.Image, label: Image.Image) -> tuple[Image.Image, Image.Image]:
  return Image.fromarray(make_8bit(aug.apply(image, transform))).convert("RGB"), Image.fromarray(mask_rgb(aug.apply(label, transform), (255, 255, 0)))
  
  
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
    

def original(_) -> Transform:
  def f(image: Image.Image) -> Image.Image:
    return image
  return f
  
  
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
  transform_builders = [original, *aug.transformations]
  transforms = [x(args.magnitude) for x in transform_builders]
  images = [x for pair in [run_transform(aug, transform, aug.images[0], aug.labels[0]) for transform in transforms] for x in pair]
  
  print("Drawing the grid...")
  titles = [x for pair in [[x.__name__ + " image", x.__name__ + " mask"] for x in transform_builders] for x in pair]
  arrange_on_grid(images, titles, args.output)
  print("Done!")