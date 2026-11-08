#!/usr/bin/env python3
"""Train a U-Net, one named invocation at a time.

Each invocation writes under ``runs/<dd-mm-YYYY>-<run_name>/``:

- ``model_epoch_<N>.safetensors`` every 10 epochs
- ``eval_scores.txt`` for that invocation only

``<dd-mm-YYYY>`` is the local date when the invocation starts (the same
format previously used for ``models/<date>/``). The date is not recomputed
per checkpoint, so one invocation stays in one directory. The same run name
on another calendar day is a different directory and is not searched.
Existing ``models/`` trees and a global ``eval_scores.txt`` are left untouched.

Resume, when ``--model-file`` is omitted, loads the highest numeric epoch in
that directory (``model_epoch_10`` beats ``model_epoch_2``). The start epoch
is the ``N`` in ``model_epoch_<N>.safetensors``, inclusive, and ``--epochs``
is still the last epoch index, not a count of extra epochs. Resuming epoch
50 with ``--epochs 100`` trains epochs 50 through 100. If ``--epochs`` is
below that start epoch, the loop does not run and the directory is left as
it is. A missing directory, an empty one, or one with no
``model_epoch_*.safetensors`` starts at epoch 1. Other files in the directory
are not removed. A leftover ``eval_scores.txt`` is replaced only once the new
invocation writes a score.

``--model-file`` still loads that exact file (including a checkpoint from
another date or from ``models/``) and still wins over the directory's latest
file. It is read before any checkpoint in the current run directory is
removed. A checkpoint outside the current run directory is never deleted.

After an invocation has saved at least one checkpoint, ``model_epoch_*.safetensors``
in its directory that this invocation did not write are removed. A shorter
follow-up therefore does not keep the previous invocation's extra checkpoints.
If the invocation saves none, nothing is deleted.
"""

from tinygrad.tensor import Tensor
from tinygrad.engine.jit import TinyJit
from tinygrad.nn.optim import Adam, Optimizer
from tinygrad.nn.state import get_parameters, load_state_dict, safe_load, safe_save, get_state_dict
from tinygrad.helpers import tqdm, Context
from tinygrad_unet.net import UNet
from tinygrad_unet.dataset import TRAIN_DATASET, VAL_DATASET, load_dataset
from tinygrad_unet.inference import load_combined_mask, infer_and_overlap
from datetime import datetime
from random import shuffle
from pathlib import Path
import argparse
import json
import glob
import re


EPOCHS = 500
SAVE_BEST_DICE_PREDICTION = False


def validate(model: UNet, dataset: list[Tensor]) -> float:
    """Validate the model by the means of calculating the DICE coefficient"""
    total_dice = 0.0
    l = int(dataset[0].shape[0])
    
    @TinyJit
    def f(example: Tensor, label: Tensor) -> Tensor:
        pred = model(example)
        probs = pred.sigmoid()
        smooth = 1e-6
        return (2.0 * (probs * label).sum() + smooth) / (probs.sum() + label.sum() + smooth)
      
    with Context(TRAINING=0):
      for idx in tqdm(range(l), desc="Validating"): total_dice += f(dataset[0][idx].clone(), dataset[1][idx].clone()).numpy()
      
    return total_dice / l


@TinyJit
def tiny_step(example: Tensor, label: Tensor, model: UNet, optimizer: Optimizer) -> Tensor:
    optimizer.zero_grad()
    logits = model(example)

    # smooth = 1e-6
    
    # Focal loss (based on https://github.com/facebookresearch/fvcore/blob/main/fvcore/nn/focal_loss.py)
    # alpha = 0.999 # This is the weight assigned to the rare class
    # gamma = 3 # This is the focusing factor which decides how important getting right positive predictions is
    p = logits.sigmoid()
    # bce = logits.binary_crossentropy_logits(label, reduction="none")
    # pt = p * label + (1 - p) * (1 - label)
    # alpha_t = alpha * label + (1 - alpha) * (1 - label)
    # focal_loss = (alpha_t * bce * (1 - pt).pow(gamma)).mean()

    # Tversky loss
    alpha = 0.3 # Weight of false-positives
    beta = 0.7 # Weight of false-negatives
    s = (p * label).sum()
    tversky_loss = 1 - s / (s + alpha * (p * (1 - label)).sum() + beta * ((1 - p) * label).sum())

    # Continuous DICE Coefficient (paper: https://www.biorxiv.org/content/10.1101/306977v1)
    # probs = logits.sigmoid()
    # intersect = (probs * label).sum()
    # c = (intersect > 0).where(intersect / ((label * probs.sign()).sum() + smooth), 1)
    # cdice = (label.sum() + probs.sum() == 0).where(0, 1 - (2 * intersect) / (c * label.sum() + probs.sum() + smooth))

    loss = tversky_loss
    loss.backward()
    optimizer.step()
    return loss


def train_epoch(model: UNet, dataset: list[Tensor], optimizer: Optimizer, beam: bool) -> Tensor:
    """Train for one epoch"""
    
    total_loss = Tensor(0.0)
    l = int(dataset[0].shape[0])
    indices = list(range(l))
    shuffle(indices)

    with Context(TRAINING=1, BEAM=2 if beam else 0):
      for i in tqdm(indices, desc="Training"):
        # Using Tensor.clone() before passing the tenors as parameters as per https://github.com/tinygrad/tinygrad/blob/master/tinygrad/engine/jit.py#L91
        example, label = dataset[0][i].clone(), dataset[1][i].clone()
        loss = tiny_step(example, label, model, optimizer)
        # TODO: Print the gradients here to check whether they are diminishing.
        total_loss += loss
    
    return total_loss / l


def choose_preview_image() -> str | None:
    """Choose the image that contains the mask"""
    with open("training_files.json") as f: examples = json.load(f)
    shuffle(examples)
    for example in examples:
        if (load_combined_mask(example).max() > 0).numpy(): return example
    return None
    

def get_latest_dataset(i: int, dataset_location: str) -> str:
  if i == 1: return dataset_location
  augumented_datasets = glob.glob(f"{dataset_location.split('.')[0]}_*.safetensors")
  if len(augumented_datasets) == 0: return dataset_location
  return sorted(augumented_datasets, key=lambda x: int(re.search(r'_(\d+)', x).group(1)))[:i-1][-1]
    

def load_augumented_dataset(i: int, dataset_location: str) -> list[Tensor]:
  dataset_file = get_latest_dataset(i, dataset_location)
  print(f'For this epoch using the dataset from "{dataset_file}".')
  return load_dataset(dataset_file)


RUNS_ROOT = Path("runs")
DATE_FORMAT = "%d-%m-%Y"


def validate_run_name(run_name: str) -> str:
  """Reject run names that are not a single path component."""
  if not isinstance(run_name, str) or run_name != run_name.strip() or run_name in {"", ".", ".."}:
    raise ValueError(f"Invalid run name {run_name!r}: expected one path component")
  if "/" in run_name or "\\" in run_name or "\0" in run_name or Path(run_name).name != run_name:
    raise ValueError(f"Invalid run name {run_name!r}: expected one path component")
  return run_name


def run_directory(run_name: str, when: datetime, runs_root: str | Path = RUNS_ROOT) -> Path:
  """``runs/<dd-mm-YYYY>-<run_name>/`` for the invocation's start date."""
  validate_run_name(run_name)
  return Path(runs_root) / f"{when.strftime(DATE_FORMAT)}-{run_name}"


def checkpoint_epoch(model_file: str | Path) -> int:
  """Epoch encoded in ``model_epoch_<N>.safetensors``.

  The previous trainer used ``int(model_file.split("_")[2][:-12])``, which
  selects ``N`` and strips the ``.safetensors`` suffix. That expression is
  applied to the file name so underscores in a run name do not shift it.
  Training resumes at this epoch, inclusive.
  """
  name = Path(model_file).name
  parts = name.split("_")
  if len(parts) < 3:
    raise ValueError(f"Cannot read an epoch from checkpoint name {name!r}")
  token = parts[2][:-12]
  if token == "" or not token.isdigit():
    raise ValueError(f"Cannot read an epoch from checkpoint name {name!r}")
  return int(token)


def latest_checkpoint(run_dir: Path) -> Path | None:
  """Highest numeric ``model_epoch_*.safetensors`` in one run directory, or none."""
  if not run_dir.is_dir(): return None
  ranked: list[tuple[int, str, Path]] = []
  for path in run_dir.glob("model_epoch_*.safetensors"):
    try: epoch = checkpoint_epoch(path)
    except ValueError: continue
    ranked.append((epoch, path.name, path))
  if not ranked: return None
  ranked.sort()
  return ranked[-1][2]


def resolve_resume_checkpoint(run_dir: Path, model_file: str | None) -> str | None:
  """Use ``--model-file`` when given; otherwise the latest checkpoint in ``run_dir``."""
  if model_file: return model_file
  latest = latest_checkpoint(run_dir)
  return str(latest) if latest is not None else None


def write_eval_score(scores_path: Path, message: str, *, replace: bool) -> None:
  scores_path.parent.mkdir(parents=True, exist_ok=True)
  with open(scores_path, "w" if replace else "a") as handle: handle.write(message)


def discard_unwritten_checkpoints(run_dir: Path, written: set[Path]) -> None:
  """Remove checkpoint files in ``run_dir`` that this invocation did not save.

  No-op when nothing was saved, so a loaded checkpoint is not deleted if this
  invocation never reaches a save step. Call only after that checkpoint has
  been loaded.
  """
  if not written or not run_dir.is_dir(): return
  kept = {path.resolve() for path in written}
  for path in run_dir.glob("model_epoch_*.safetensors"):
    try: checkpoint_epoch(path)
    except ValueError: continue
    if path.resolve() not in kept: path.unlink()


def run_training(val: list[Tensor], epochs: int, model_file: str | None, beam: bool, pred_basedir: str = "predictions", dataset_location: str = TRAIN_DATASET, *, run_name: str, runs_root: str | Path = RUNS_ROOT, started_at: datetime | None = None):
  when = started_at if started_at is not None else datetime.now()
  run_dir = run_directory(run_name, when, runs_root)
  # Resolve and load before any score replacement or checkpoint deletion.
  resume_from = resolve_resume_checkpoint(run_dir, model_file)
  print(f"Run directory: {run_dir}")
  model = UNet()
  if resume_from:
    state = safe_load(resume_from)  # Load model checkpoint
    load_state_dict(model, state)
    print(f"Loaded model from {resume_from}.")
  optim = Adam(get_parameters(model), 1e-4)
  preview_image = choose_preview_image()
  if preview_image is None:
    print("All masks are empty.")
    return
  dice = 0
  largest_dice = 0
  start_epoch = 1 if resume_from is None else checkpoint_epoch(resume_from)
  written: set[Path] = set()
  scores_path = run_dir / "eval_scores.txt"
  wrote_score = False
  for i in range(start_epoch, epochs+1):
    epoch_msg = f"\nEpoch {i}/{epochs}"
    print(epoch_msg)
    train = load_augumented_dataset(i, dataset_location)
    train_loss = train_epoch(model, train, optim, beam)
    train_loss_msg = f"Train Loss: {train_loss.numpy():.6f}"
    print(train_loss_msg)
    dice = validate(model, val)
    val_msg = ", ".join([epoch_msg, train_loss_msg, f"Epoch Validation Dice: {dice:.6f}"])
    print(val_msg)
    write_eval_score(scores_path, val_msg, replace=not wrote_score)
    wrote_score = True
    if i % 10 == 0:
        with Context(TRAINING=0): infer_and_overlap(model, preview_image, pred_basedir + "/training_validation", i)
    if i % 10 == 0:
        print("Saving the model...")
        checkpoint = run_dir / f"model_epoch_{i}.safetensors"
        checkpoint.parent.mkdir(parents=True, exist_ok=True)
        safe_save(get_state_dict(model), str(checkpoint))
        written.add(checkpoint)
        print("Saved.")
    if dice > largest_dice:
        print("This is the best DICE so far!!!")
        largest_dice = dice
        if SAVE_BEST_DICE_PREDICTION:
            with Context(TRAINING=0): infer_and_overlap(model, preview_image, pred_basedir + "/best_dice", i)
  discard_unwritten_checkpoints(run_dir, written)


def build_parser() -> argparse.ArgumentParser:
  argument_parser = argparse.ArgumentParser(
    prog="Training script",
    description="""
      This is the training script. This script loads the validation and training datasets from a default location. It expects to use a different compiled dataset for each epoch but can fall back to the dataset for the latest available epoch if dataset for the current epoch is unavailable.
    """
  )
  argument_parser.add_argument("run_name", help="Single path component. Artifacts go to runs/<dd-mm-YYYY>-<run_name>/.")
  argument_parser.add_argument("-e", "--epochs", type=int)
  argument_parser.add_argument("-m", "--model-file")
  argument_parser.add_argument("-b", "--beam", type=bool, default=True)
  return argument_parser


def main(argv: list[str] | None = None) -> None:
  args = build_parser().parse_args(argv)
  validate_run_name(args.run_name)
  print(f"Loading validation dataset from {VAL_DATASET}...")
  val = load_dataset(VAL_DATASET)
  print("Dataset loaded.")
  run_training(val, args.epochs if args.epochs is not None else EPOCHS, args.model_file, args.beam, run_name=args.run_name)


if __name__ == "__main__":
  main()
