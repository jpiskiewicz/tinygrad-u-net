#!/usr/bin/env python3

"""
This script loads augumented datasets from a chosen directory and arranges image-mask pairs
at the chosen index in a grid. This allows the user to inspect whether the transformations
worked as expected.
"""

from argparse import ArgumentParser
from pathlib import Path
from PIL import Image, ImageDraw
from tinygrad_unet.dataset import load_dataset
from test_transforms import convert_to_image, GRID_GAP, PADDING_TOP, FONT
from glob import glob


type Datapoint = tuple[Image.Image, Image.Image]


def unpack_datapoint(dataset_path: str, i: int) -> Datapoint:
  print(f'Loading image and label for datapoint number {i} from "{dataset_path}"')
  dataset = load_dataset(dataset_path)
  return convert_to_image((dataset[0][i], dataset[1][i]))


def arrange_on_grid(datapoints: list[Datapoint], index: int, output_path: Path, directory: Path):
  image_width, image_height = datapoints[0][0].size
  cols = len(datapoints)
  grid_width, grid_height = image_width * cols + (GRID_GAP * (cols + 1)), (image_height + GRID_GAP) * 2
  res = Image.new("RGB", (grid_width, grid_height + PADDING_TOP), color="white")
  for i in range(cols):
    for j in range(2):
      print(f"Drawing image {i*2 + j + 1} of {cols*2}")
      res.paste(datapoints[i][j], (image_width * i + (GRID_GAP * (i + 1)), PADDING_TOP + (image_height + GRID_GAP) * j))
  ImageDraw.Draw(res).text((grid_width / 2, PADDING_TOP / 2), f"{index} element out of each dataset found in {directory}", (0, 0, 0), FONT, "mm")
  res.save(output_path)


if __name__ == "__main__":
  parser = ArgumentParser(
    prog="inspect_augumented_dataset",
    description="""
    This script loads augumented datasets from a chosen directory and arranges image-mask pairs
    at the chosen index i in a grid. This allows the user to inspect whether the transformations
    worked as expected.
    """
  )
  parser.add_argument("directory", type=Path)
  parser.add_argument("-i", type=int, default=0)
  args = parser.parse_args()
  datapoints = [unpack_datapoint(f, args.i) for f in sorted(glob(str(args.directory) + "/*.safetensors"))]
  arrange_on_grid(datapoints, args.i, args.directory / "inspect_augumented_datasets.png", args.directory)
  