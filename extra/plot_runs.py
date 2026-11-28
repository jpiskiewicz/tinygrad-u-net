#!/usr/bin/env python3

from argparse import ArgumentParser
from pathlib import Path
from typing import cast
import plotille

"""
This tool takes in arbitrary number of paths to train runs and creates a plot
with train loss values from `eval_scores.txt` for each run.
"""


type DataPoints = tuple[list[int], list[float]]


def extract_run_loss(path: Path) -> DataPoints:
  res: DataPoints = ([], [])
  with open(path / "eval_scores.txt") as f:
    lines = f.readlines()[:-1]
    for line in lines:
      cols = line.split(", ")
      res[0].append(int(cols[0].split(" ")[1].split("/")[0]))
      res[1].append(float(cols[1].split(" ")[2]))
  return res


if __name__ == "__main__":
  parser = ArgumentParser()
  parser.add_argument("paths", type=Path, nargs="+", help="list of directories containing train runs")
  args = parser.parse_args()
  runs = [(path, extract_run_loss(path)) for path in args.paths]
  fig = plotille.Figure()
  fig.width = 60
  fig.height = 30
  fig.set_x_limits(min_=0, max_=max(x for run in runs for x in run[1][0]))
  fig.set_y_limits(min_=0, max_=1)
  for run in runs:
    fig.plot(run[1][0], run[1][1], label=run[0])
  print(fig.show(legend=True))