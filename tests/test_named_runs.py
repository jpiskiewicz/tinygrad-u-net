#!/usr/bin/env python3
"""Named-run paths, resume loading, and score replacement.

Training steps are mocked. Checkpoint save/load and the training entrypoint
are the production implementations.
"""

import os
import unittest
from datetime import datetime
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import numpy as np
from tinygrad.nn.state import get_state_dict, safe_load, safe_save
from tinygrad.tensor import Tensor

from scripts.train import (
  build_parser,
  checkpoint_epoch,
  main,
  run_training,
  validate_run_name,
)


WHEN = datetime(2026, 11, 8)
DATE_NAME = "08-11-2026"


class ProbeNet:
  def __init__(self):
    self.weight = Tensor([0.0])


def state_of(model) -> dict[str, np.ndarray]:
  return {key: np.array(value.numpy(), copy=True) for key, value in get_state_dict(model).items()}


def file_state(path: Path) -> dict[str, np.ndarray]:
  return {key: np.array(value.numpy(), copy=True) for key, value in safe_load(path).items()}


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
      if not path.is_file():
        raise AssertionError(f"resume checkpoint missing before load: {fn}")
      self.loads.append(str(path))
      return safe_load(fn)

    self._patches = [
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

  def invoke(self, run_name: str, epochs: int, model_file: str | None = None, started_at: datetime = WHEN, runs_root: Path | None = None):
    self.seen = []
    self.loads = []
    kwargs = {"run_name": run_name, "started_at": started_at}
    if runs_root is not None: kwargs["runs_root"] = runs_root
    run_training([Tensor([0.0]), Tensor([0.0])], epochs, model_file, False, str(self.root / "predictions"), "unused", **kwargs)
    return self.seen

  def test_fresh_run_writes_date_name_directory_and_no_global_score(self):
    os.chdir(self.root)
    seen = self.invoke("alpha", 10)
    run_dir = self.root / "runs" / f"{DATE_NAME}-alpha"
    self.assertEqual(weight_of(seen[0]), 0.0)
    self.assertEqual(self.loads, [])
    self.assertTrue((run_dir / "model_epoch_10.safetensors").is_file())
    self.assertEqual(weight_of(file_state(run_dir / "model_epoch_10.safetensors")), 10.0)
    scores = (run_dir / "eval_scores.txt").read_text()
    self.assertIn("\nEpoch 1/10", scores)
    self.assertIn("\nEpoch 10/10", scores)
    self.assertFalse((self.root / "eval_scores.txt").exists())
    self.assertFalse((self.root / "models").exists())

  def test_same_date_different_names_are_isolated(self):
    os.chdir(self.root)
    seen_alpha = self.invoke("alpha", 10)
    seen_beta = self.invoke("beta", 10)
    self.assertEqual(weight_of(seen_alpha[0]), 0.0)
    self.assertEqual(weight_of(seen_beta[0]), 0.0)
    alpha = self.root / "runs" / f"{DATE_NAME}-alpha"
    beta = self.root / "runs" / f"{DATE_NAME}-beta"
    beta_checkpoint = (beta / "model_epoch_10.safetensors").read_bytes()
    beta_scores = (beta / "eval_scores.txt").read_bytes()
    seen = self.invoke("alpha", 20)
    self.assertEqual(Path(self.loads[0]).name, "model_epoch_10.safetensors")
    self.assertEqual(weight_of(seen[0]), 10.0)
    text = (alpha / "eval_scores.txt").read_text()
    self.assertNotIn("Epoch 1/", text)
    self.assertIn("\nEpoch 10/20", text)
    self.assertIn("\nEpoch 20/20", text)
    self.assertEqual(sorted(path.name for path in alpha.glob("model_epoch_*.safetensors")), ["model_epoch_10.safetensors", "model_epoch_20.safetensors"])
    self.assertEqual((beta / "model_epoch_10.safetensors").read_bytes(), beta_checkpoint)
    self.assertEqual((beta / "eval_scores.txt").read_bytes(), beta_scores)

  def test_shorter_resume_loads_previous_weights_and_drops_stale_outputs(self):
    os.chdir(self.root)
    legacy = plant_checkpoint(self.root / "models" / DATE_NAME, 10, 7.0)
    global_scores = self.root / "eval_scores.txt"
    global_scores.write_text("GLOBAL-SENTINEL")
    other = self.root / "runs" / "01-01-2000-other"
    other_checkpoint = plant_checkpoint(other, 4, 4.0)
    (other / "eval_scores.txt").write_text("OTHER-SCORES")
    yesterday_dir = self.root / "runs" / "07-11-2026-cell"
    yesterday = plant_checkpoint(yesterday_dir, 10, 123.0)
    (yesterday_dir / "eval_scores.txt").write_text("YESTERDAY")
    sibling_dir = self.root / "runs" / f"{DATE_NAME}-other"
    sibling = plant_checkpoint(sibling_dir, 10, 8.0)
    (sibling_dir / "eval_scores.txt").write_text("SIBLING")
    preserved = {
      legacy: legacy.read_bytes(),
      global_scores: global_scores.read_bytes(),
      other_checkpoint: other_checkpoint.read_bytes(),
      other / "eval_scores.txt": (other / "eval_scores.txt").read_bytes(),
      yesterday: yesterday.read_bytes(),
      yesterday_dir / "eval_scores.txt": (yesterday_dir / "eval_scores.txt").read_bytes(),
      sibling: sibling.read_bytes(),
      sibling_dir / "eval_scores.txt": (sibling_dir / "eval_scores.txt").read_bytes(),
    }

    first = self.invoke("cell", 30)
    self.assertEqual(weight_of(first[0]), 0.0)
    self.assertEqual(self.loads, [])
    run_dir = self.root / "runs" / f"{DATE_NAME}-cell"
    self.assertEqual(sorted(path.name for path in run_dir.glob("model_epoch_*.safetensors")), ["model_epoch_10.safetensors", "model_epoch_20.safetensors", "model_epoch_30.safetensors"])
    first_scores = (run_dir / "eval_scores.txt").read_text()
    self.assertIn("\nEpoch 1/30", first_scores)
    self.assertIn("\nEpoch 30/30", first_scores)
    before = file_state(run_dir / "model_epoch_30.safetensors")
    self.assertEqual(weight_of(before), 30.0)
    (run_dir / "notes.txt").write_text("keep-me")

    seen = self.invoke("cell", 30)
    self.assertEqual(Path(self.loads[0]).name, "model_epoch_30.safetensors")
    np.testing.assert_array_equal(seen[0]["weight"], before["weight"])
    self.assertEqual(sorted(path.name for path in run_dir.glob("model_epoch_*.safetensors")), ["model_epoch_30.safetensors"])
    second_scores = (run_dir / "eval_scores.txt").read_text()
    self.assertNotIn("Epoch 1/", second_scores)
    self.assertNotIn("Epoch 10/", second_scores)
    self.assertNotIn("Epoch 20/", second_scores)
    self.assertEqual(second_scores.count("Epoch 30/30"), 1)
    self.assertNotIn(first_scores.strip(), second_scores)
    self.assertEqual(weight_of(file_state(run_dir / "model_epoch_30.safetensors")), 31.0)
    self.assertEqual((run_dir / "notes.txt").read_text(), "keep-me")
    for path, blob in preserved.items():
      self.assertEqual(path.read_bytes(), blob, path)
    self.assertEqual(global_scores.read_text(), "GLOBAL-SENTINEL")

  def test_latest_checkpoint_is_numeric_not_lexical(self):
    os.chdir(self.root)
    run_dir = self.root / "runs" / f"{DATE_NAME}-numeric"
    plant_checkpoint(run_dir, 2, 2.0)
    plant_checkpoint(run_dir, 10, 10.0)
    names = sorted(path.name for path in run_dir.glob("model_epoch_*.safetensors"))
    self.assertEqual(names[-1], "model_epoch_2.safetensors")
    seen = self.invoke("numeric", 10)
    self.assertEqual(Path(self.loads[0]).name, "model_epoch_10.safetensors")
    self.assertEqual(weight_of(seen[0]), 10.0)
    self.assertEqual(len(seen), 1)

  def test_empty_directory_starts_at_epoch_one(self):
    os.chdir(self.root)
    run_dir = self.root / "runs" / f"{DATE_NAME}-empty"
    run_dir.mkdir(parents=True)
    (run_dir / "notes.txt").write_text("keep")
    (run_dir / "eval_scores.txt").write_text("OLD-SCORE")
    seen = self.invoke("empty", 2)
    self.assertEqual(self.loads, [])
    self.assertEqual(weight_of(seen[0]), 0.0)
    self.assertEqual(len(seen), 2)
    text = (run_dir / "eval_scores.txt").read_text()
    self.assertIn("\nEpoch 1/2", text)
    self.assertIn("\nEpoch 2/2", text)
    self.assertNotIn("OLD-SCORE", text)
    self.assertEqual((run_dir / "notes.txt").read_text(), "keep")
    self.assertEqual(list(run_dir.glob("model_epoch_*.safetensors")), [])

  def test_unparseable_checkpoint_name_is_not_loaded_or_deleted(self):
    os.chdir(self.root)
    run_dir = self.root / "runs" / f"{DATE_NAME}-bogus"
    run_dir.mkdir(parents=True)
    bogus = run_dir / "model_epoch_x.safetensors"
    bogus.write_bytes(b"not a tensor")
    seen = self.invoke("bogus", 10)
    self.assertEqual(self.loads, [])
    self.assertEqual(weight_of(seen[0]), 0.0)
    self.assertTrue((run_dir / "model_epoch_10.safetensors").is_file())
    self.assertEqual(bogus.read_bytes(), b"not a tensor")

  def test_other_date_same_name_is_not_resumed(self):
    os.chdir(self.root)
    yesterday_dir = self.root / "runs" / "07-11-2026-cell"
    checkpoint = plant_checkpoint(yesterday_dir, 30, 123.0)
    (yesterday_dir / "eval_scores.txt").write_text("YESTERDAY")
    checkpoint_bytes = checkpoint.read_bytes()
    score_bytes = (yesterday_dir / "eval_scores.txt").read_bytes()
    seen = self.invoke("cell", 1)
    self.assertEqual(self.loads, [])
    self.assertEqual(weight_of(seen[0]), 0.0)
    self.assertEqual(checkpoint.read_bytes(), checkpoint_bytes)
    self.assertEqual((yesterday_dir / "eval_scores.txt").read_bytes(), score_bytes)
    today = (self.root / "runs" / f"{DATE_NAME}-cell" / "eval_scores.txt").read_text()
    self.assertIn("\nEpoch 1/1", today)

  def test_explicit_model_file_sets_inclusive_epoch_and_keeps_external_file(self):
    os.chdir(self.root)
    external = plant_checkpoint(self.root / "outside", 15, 15.0)
    external_bytes = external.read_bytes()
    run_dir = self.root / "runs" / f"{DATE_NAME}-explicit"
    plant_checkpoint(run_dir, 40, 40.0)
    seen = self.invoke("explicit", 20, model_file=str(external))
    self.assertEqual(Path(self.loads[0]).resolve(), external.resolve())
    self.assertEqual(weight_of(seen[0]), 15.0)
    self.assertEqual(len(seen), 6)
    text = (run_dir / "eval_scores.txt").read_text()
    self.assertIn("\nEpoch 15/20", text)
    self.assertIn("\nEpoch 20/20", text)
    self.assertNotIn("Epoch 1/", text)
    self.assertNotIn("Epoch 14/", text)
    self.assertEqual(external.read_bytes(), external_bytes)
    self.assertEqual(sorted(path.name for path in run_dir.glob("model_epoch_*.safetensors")), ["model_epoch_20.safetensors"])

  def test_epochs_below_resume_epoch_leaves_checkpoints_and_scores(self):
    os.chdir(self.root)
    run_dir = self.root / "runs" / f"{DATE_NAME}-held"
    plant_checkpoint(run_dir, 30, 30.0)
    plant_checkpoint(run_dir, 10, 10.0)
    (run_dir / "eval_scores.txt").write_text("KEEP-SCORES")
    before = {path.name: path.read_bytes() for path in run_dir.iterdir() if path.is_file()}
    seen = self.invoke("held", 10)
    self.assertEqual(Path(self.loads[0]).name, "model_epoch_30.safetensors")
    self.assertEqual(weight_of(seen[0]) if seen else None, None)
    self.assertEqual(seen, [])
    after = {path.name: path.read_bytes() for path in run_dir.iterdir() if path.is_file()}
    self.assertEqual(after, before)

  def test_resume_without_new_checkpoint_keeps_the_loaded_file(self):
    os.chdir(self.root)
    run_dir = self.root / "runs" / f"{DATE_NAME}-partial"
    checkpoint = plant_checkpoint(run_dir, 15, 15.0)
    before = checkpoint.read_bytes()
    seen = self.invoke("partial", 19)
    self.assertEqual(Path(self.loads[0]).name, "model_epoch_15.safetensors")
    self.assertEqual(weight_of(seen[0]), 15.0)
    self.assertEqual(len(seen), 5)
    self.assertEqual(checkpoint.read_bytes(), before)
    self.assertEqual([path.name for path in run_dir.glob("model_epoch_*.safetensors")], ["model_epoch_15.safetensors"])
    text = (run_dir / "eval_scores.txt").read_text()
    self.assertIn("\nEpoch 15/19", text)
    self.assertIn("\nEpoch 19/19", text)
    self.assertNotIn("Epoch 1/", text)

  def test_checkpoint_epoch_uses_filename_not_directory_underscores(self):
    for epoch in (1, 2, 10, 50, 100):
      name = f"model_epoch_{epoch}.safetensors"
      self.assertEqual(checkpoint_epoch(name), int(name.split("_")[2][:-12]))
      self.assertEqual(checkpoint_epoch(f"runs/{DATE_NAME}-my_run/{name}"), epoch)
    with self.assertRaises(ValueError):
      checkpoint_epoch("not-a-checkpoint.safetensors")

  def test_cli_positional_parsing_and_invalid_run_names(self):
    parser = build_parser()
    args = parser.parse_args(["exp", "-e", "50", "-m", "weights.safetensors"])
    self.assertEqual(args.run_name, "exp")
    self.assertEqual(args.epochs, 50)
    self.assertEqual(args.model_file, "weights.safetensors")
    bare = parser.parse_args(["exp"])
    self.assertEqual(bare.run_name, "exp")
    self.assertIsNone(bare.epochs)
    self.assertIsNone(bare.model_file)
    self.assertTrue(bare.beam)
    with self.assertRaises(SystemExit):
      parser.parse_args([])
    with self.assertRaises(SystemExit):
      parser.parse_args(["-e", "3"])

    for bad in ["", " ", ".", "..", "a/b", "a\\b", "../escape", "foo/../../etc", "a\x00b", " padded"]:
      with self.assertRaises(ValueError):
        validate_run_name(bad)
    for good in ["exp", "exp_a.1", "exp-a", "Experiment 1"]:
      self.assertEqual(validate_run_name(good), good)

    called = []

    def boom(path):
      called.append(path)
      raise AssertionError("dataset loaded")

    with patch("scripts.train.load_dataset", boom), self.assertRaises(ValueError):
      main(["../escape", "-e", "1"])
    self.assertEqual(called, [])
    with self.assertRaises(ValueError):
      run_training([], 1, None, False, run_name="../escape", runs_root=self.root, started_at=WHEN)
    self.assertEqual(list(self.root.iterdir()), [])


if __name__ == "__main__":
  unittest.main()
