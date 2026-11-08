#!/usr/bin/env python3

import os
import unittest
from datetime import datetime
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import numpy as np
from tinygrad.nn.state import get_state_dict, safe_load, safe_save
from tinygrad.tensor import Tensor

from scripts.train import run_training


WHEN = datetime(2026, 11, 8)
DATE_NAME = "08-11-2026"


class FrozenDate:
  @staticmethod
  def now(): return WHEN


class ProbeNet:
  def __init__(self):
    self.weight = Tensor([0.0])


def state_of(model) -> dict[str, np.ndarray]:
  return {key: np.array(value.numpy(), copy=True) for key, value in get_state_dict(model).items()}


def file_weight(path: Path) -> float:
  return float(np.array(safe_load(path)["weight"].numpy()).reshape(-1)[0])


def plant_checkpoint(directory: Path, epoch: int, value: float) -> Path:
  directory.mkdir(parents=True, exist_ok=True)
  net = ProbeNet()
  net.weight.assign(Tensor([value])).realize()
  path = directory / f"model_epoch_{epoch}.safetensors"
  safe_save(get_state_dict(net), str(path))
  return path


def weight_of(state: dict[str, np.ndarray]) -> float:
  return float(np.array(state["weight"]).reshape(-1)[0])


class NamedRunTests(unittest.TestCase):
  def setUp(self):
    self._cwd = os.getcwd()
    self.tmp = TemporaryDirectory()
    self.root = Path(self.tmp.name)
    self.seen: list[dict[str, np.ndarray]] = []
    self.loads: list[str] = []

    def train_epoch(model, dataset, optimizer, beam):
      self.seen.append(state_of(model))
      value = weight_of(self.seen[-1]) + 1.0
      model.weight.assign(Tensor([value])).realize()
      return Tensor(0.25)

    def validate(model, dataset):
      return weight_of(state_of(model))

    def wrapped_load(fn):
      path = Path(fn)
      if not path.is_file(): raise AssertionError(f"checkpoint missing before load: {fn}")
      self.loads.append(str(path))
      return safe_load(fn)

    self._patches = [
      patch("scripts.train.datetime", FrozenDate),
      patch("scripts.train.UNet", ProbeNet),
      patch("scripts.train.train_epoch", train_epoch),
      patch("scripts.train.validate", validate),
      patch("scripts.train.choose_preview_image", lambda: "preview"),
      patch("scripts.train.load_augumented_dataset", lambda i, dataset_location: [Tensor([0.0]), Tensor([0.0])]),
      patch("scripts.train.infer_and_overlap", lambda *args, **kwargs: None),
      patch("scripts.train.safe_load", wrapped_load),
    ]
    for item in self._patches: item.start()
    self.addCleanup(self._stop_patches)
    self.addCleanup(self.tmp.cleanup)
    self.addCleanup(lambda: os.chdir(self._cwd))

  def _stop_patches(self):
    for item in reversed(self._patches): item.stop()

  def invoke(self, run_name: str, epochs: int, model_file: str | None = None):
    self.seen = []
    self.loads = []
    run_training([Tensor([0.0]), Tensor([0.0])], epochs, model_file, False, str(self.root / "predictions"), "unused", self.root / "runs", run_name=run_name)
    return self.seen

  def test_repeat_same_name_without_model_file_starts_fresh_and_replaces_outputs(self):
    other = plant_checkpoint(self.root / "runs" / f"{DATE_NAME}-other", 10, 8.0)
    (other.parent / "eval_scores.txt").write_text("OTHER-SCORES")
    yesterday = plant_checkpoint(self.root / "runs" / "01-01-2000-cell", 30, 123.0)
    (yesterday.parent / "eval_scores.txt").write_text("YESTERDAY")
    legacy = plant_checkpoint(self.root / "models" / DATE_NAME, 10, 7.0)
    global_scores = self.root / "eval_scores.txt"
    global_scores.write_text("GLOBAL-SENTINEL")
    preserved = [other, other.parent / "eval_scores.txt", yesterday, yesterday.parent / "eval_scores.txt", legacy, global_scores]
    before = {path: path.read_bytes() for path in preserved}

    first = self.invoke("cell", 30)
    self.assertEqual(self.loads, [])
    self.assertEqual(weight_of(first[0]), 0.0)
    run_dir = self.root / "runs" / f"{DATE_NAME}-cell"
    self.assertEqual(sorted(path.name for path in run_dir.glob("model_epoch_*.safetensors")), ["model_epoch_10.safetensors", "model_epoch_20.safetensors", "model_epoch_30.safetensors"])
    self.assertEqual(file_weight(run_dir / "model_epoch_30.safetensors"), 30.0)
    first_scores = (run_dir / "eval_scores.txt").read_text()
    self.assertIn("\nEpoch 1/30", first_scores)
    self.assertIn("\nEpoch 30/30", first_scores)
    self.assertFalse((self.root / "models" / DATE_NAME / "eval_scores.txt").exists())

    second = self.invoke("cell", 10)
    self.assertEqual(self.loads, [])
    self.assertEqual(weight_of(second[0]), 0.0)
    self.assertEqual(sorted(path.name for path in run_dir.glob("model_epoch_*.safetensors")), ["model_epoch_10.safetensors", "model_epoch_20.safetensors", "model_epoch_30.safetensors"])
    self.assertEqual(file_weight(run_dir / "model_epoch_10.safetensors"), 10.0)
    self.assertEqual(file_weight(run_dir / "model_epoch_20.safetensors"), 20.0)
    self.assertEqual(file_weight(run_dir / "model_epoch_30.safetensors"), 30.0)
    second_scores = (run_dir / "eval_scores.txt").read_text()
    self.assertIn("\nEpoch 1/10", second_scores)
    self.assertIn("\nEpoch 10/10", second_scores)
    self.assertNotIn("Epoch 11/", second_scores)
    self.assertNotIn("Epoch 30/", second_scores)
    self.assertNotIn("/30", second_scores)
    for path, blob in before.items(): self.assertEqual(path.read_bytes(), blob, path)
    self.assertEqual(global_scores.read_text(), "GLOBAL-SENTINEL")

  def test_explicit_model_file_loads_requested_weights(self):
    external = plant_checkpoint(self.root / "outside", 15, 15.0)
    external_bytes = external.read_bytes()
    run_dir = self.root / "runs" / f"{DATE_NAME}-explicit"
    plant_checkpoint(run_dir, 40, 40.0)
    (run_dir / "eval_scores.txt").write_text("OLD-SCORES")
    seen = self.invoke("explicit", 20, model_file=str(external))
    self.assertEqual(Path(self.loads[0]).resolve(), external.resolve())
    self.assertEqual(weight_of(seen[0]), 15.0)
    self.assertEqual(len(seen), 5)
    text = (run_dir / "eval_scores.txt").read_text()
    self.assertIn("\nEpoch 16/20", text)
    self.assertIn("\nEpoch 20/20", text)
    self.assertNotIn("Epoch 15/", text)
    self.assertNotIn("Epoch 1/", text)
    self.assertNotIn("OLD-SCORES", text)
    self.assertEqual(external.read_bytes(), external_bytes)
    self.assertEqual(sorted(path.name for path in run_dir.glob("model_epoch_*.safetensors")), ["model_epoch_20.safetensors", "model_epoch_40.safetensors"])
    self.assertEqual(file_weight(run_dir / "model_epoch_20.safetensors"), 20.0)
    self.assertEqual(file_weight(run_dir / "model_epoch_40.safetensors"), 40.0)

  def test_model_file_inside_run_dir_is_loaded_before_it_is_replaced(self):
    run_dir = self.root / "runs" / f"{DATE_NAME}-inside"
    requested = plant_checkpoint(run_dir, 10, 10.0)
    plant_checkpoint(run_dir, 20, 99.0)
    (run_dir / "eval_scores.txt").write_text("PRIOR")
    requested_bytes = requested.read_bytes()
    seen = self.invoke("inside", 20, model_file=str(requested))
    self.assertEqual(Path(self.loads[0]).resolve(), requested.resolve())
    self.assertEqual(requested.read_bytes(), requested_bytes)
    self.assertEqual(weight_of(seen[0]), 10.0)
    self.assertEqual(len(seen), 10)
    self.assertEqual(sorted(path.name for path in run_dir.glob("model_epoch_*.safetensors")), ["model_epoch_10.safetensors", "model_epoch_20.safetensors"])
    self.assertEqual(file_weight(run_dir / "model_epoch_20.safetensors"), 20.0)
    text = (run_dir / "eval_scores.txt").read_text()
    self.assertIn("\nEpoch 11/20", text)
    self.assertIn("\nEpoch 20/20", text)
    self.assertNotIn("Epoch 10/", text)
    self.assertNotIn("Epoch 1/", text)
    self.assertNotIn("PRIOR", text)


if __name__ == "__main__":
  unittest.main()
