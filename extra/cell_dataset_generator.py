#!/usr/bin/env python3

from PIL import Image, ImageDraw
import random

"""
This is a generator that creates a simple simulation of a dataset containing
cell images seen under an optical microscope.
The dataset contains images along with their corresponding mask files.
The dataset contains empty examples as well in order to decrease the count
of false-positive predictions.
"""


type color = tuple[int, int, int]

IMAGE_SIZE = 240
BACKGROUND_COLOUR = (181, 158, 184)
FEATURE_RADIUS = 3
FEATURE_COUNT = 100
FEATURE_COLOUR = (241, 238, 241)
DATASET_SIZE = 300
OUTPUT_DIR = "generated"


def draw_features(feature_positions: list[list[int]], color: color, background: color) -> Image.Image:
  im = Image.new("RGB", (IMAGE_SIZE, IMAGE_SIZE), color=background)
  draw = ImageDraw.Draw(im)
  for pos in feature_positions: draw.circle(pos, FEATURE_RADIUS, fill=color)
  return im


if __name__ == "__main__":
  for i in range(DATASET_SIZE):
    feature_positions = [[random.randrange(IMAGE_SIZE) for _ in range(2)] for _ in range(FEATURE_COUNT)]
    image = draw_features(feature_positions, FEATURE_COLOUR, BACKGROUND_COLOUR)
    mask = draw_features(feature_positions, (255, 255, 255), (0, 0, 0))
    image.save(f"{OUTPUT_DIR}/example ({i}).png")
    mask.save(f"{OUTPUT_DIR}/example ({i})_mask.png")