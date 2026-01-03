#!/usr/bin/env python3

from net import UNet
from tinygrad.tensor import Tensor
from tinygrad.engine.jit import TinyJit
from tinygrad.nn.optim import Adam, Optimizer
from tinygrad.nn.state import get_parameters, load_state_dict, safe_load, safe_save, get_state_dict
from tinygrad.helpers import tqdm
from inference import load_combined_mask, infer_and_overlap
from dataset import TRAIN_DATASET, VAL_DATASET
from datetime import datetime
from random import shuffle
from pathlib import Path
from sys import argv
import json


EPOCHS = 500


def validate(model: UNet, dataset: list[Tensor]) -> float:
    """Validate the model by the means of calculating the DICE coefficient"""
    total_dice = 0.0
    l = int(dataset[0].shape[0])
    
    @TinyJit
    def f(idx: int) -> Tensor:
        pred = model(dataset[0][idx])
        probs = pred.sigmoid()
        label = dataset[1][idx]
        smooth = 1e-6
        return (2.0 * (probs * label).sum() + smooth) / (probs.sum() + label.sum() + smooth)
      
    with Tensor.train(False):
      for idx in tqdm(range(l), desc="Validating"): total_dice += f(idx).numpy()
      
    return total_dice / l


@TinyJit
def tiny_step(idx: int, dataset: list[Tensor], model: UNet, optimizer: Optimizer) -> Tensor:
    optimizer.zero_grad()
    logits = model(dataset[0][idx])
    label = dataset[1][idx]

    smooth = 1e-6
    
    # Focal loss (based on https://github.com/facebookresearch/fvcore/blob/main/fvcore/nn/focal_loss.py)
    # gamma = 2.0
    # alpha = 0.25 # From the table 1 in the focal loss paper
    # p = logits.sigmoid()
    # bce = logits.binary_crossentropy_logits(label, reduction="none")
    # pt = p * label + (1 - p) * (1 - label)
    # alpha_t = alpha * label + (1 - alpha) * (1 - label)
    # focal_loss = (alpha_t * bce * (1 - pt).pow(gamma)).mean()

    # Tversky loss
    # probs = logits.sigmoid()
    # alpha_t = 0.6
    # beta_t = 0.4
    # gamma_t = 4/3
    # tp = (probs * label).sum()
    # fp = ((1.0 - label) * probs).sum()
    # fn = (label * (1.0 - probs)).sum()
    # focal_tversky_loss = (1.0 - (tp + smooth) / (tp + alpha_t * fp + beta_t * fn + smooth)).pow(gamma_t) # TODO: Change it to non-focal

    # Continuous Dice Coefficient (paper: https://www.biorxiv.org/content/10.1101/306977v1)
    probs = logits.sigmoid()
    intersect = (probs * label).sum()
    c = (intersect > 0).where(intersect / ((label * probs.sign()).sum() + smooth), 1)
    cdice = (label.sum() + probs.sum() == 0).where(0, 1 - (2 * intersect) / (c * label.sum() + probs.sum() + smooth))

    loss = 0.5 * logits.binary_crossentropy_logits(label) + 0.5 * cdice
    loss.backward()
    optimizer.step()
    return loss


def train_epoch(model: UNet, dataset: list[Tensor], optimizer: Optimizer) -> float:
    """Train for one epoch"""
    total_loss = 0.0
    l = int(dataset[0].shape[0])
    
    indices = list(range(l))
    shuffle(indices)
    
    with Tensor.train(True):
      for idx in tqdm(indices, desc="Training"):
        loss = tiny_step(idx, dataset, model, optimizer)
        # TODO: Print the gradients here to check whether they are diminishing.
        total_loss += loss.numpy()
    
    return total_loss / l


def choose_preview_image() -> str | None:
    """Choose the image that contains the mask"""
    with open("training_files.json") as f: examples = json.load(f)
    shuffle(examples)
    for example in examples:
        if (load_combined_mask(example).max() > 0).numpy(): return example
    return None
    

def run_training(train: list[Tensor], val: list[Tensor], model_file: str | None):
  model = UNet()
  if model_file:
    state = safe_load(model_file)  # Load model checkpoint
    load_state_dict(model, state)
    print(f"Loaded model from {model_file}.")
  optim = Adam(get_parameters(model), 1e-4)
  preview_image = choose_preview_image()
  if preview_image is None:
    print("All masks are empty.")
    return
  dice = 0
  largest_dice = 0
  epochs = int(argv[2]) if len(argv) == 3 else EPOCHS
  for i in range(1 if len(argv) == 1 else int(argv[1].split("_")[2][:-12]), epochs+1):
    epoch_msg = f"\nEpoch {i}/{epochs}"
    print(epoch_msg)
    train_loss = train_epoch(model, train, optim)
    train_loss_msg = f"Train Loss: {train_loss:.6f}"
    print(train_loss_msg)
    dice = validate(model, val)
    val_msg = ", ".join([epoch_msg, train_loss_msg, f"Epoch Validation Dice: {dice:.6f}"])
    print(val_msg)
    with open("eval_scores.txt", "a") as f: f.write(val_msg)
    if i % 10 == 0:
        with Tensor.train(False): infer_and_overlap(model, preview_image, "training_validation", i)
    if i % 20 == 0:
        print("Saving the model...")
        p = Path(f"models/{datetime.now().strftime('%d-%m-%Y')}/model_epoch_{i}.safetensors")
        p.parent.mkdir(parents=True, exist_ok=True)
        safe_save(get_state_dict(model), str(p))
        print("Saved.")
    if dice > largest_dice:
        print("This is the best DICE so far!!!")
        largest_dice = dice
        with Tensor.train(False): infer_and_overlap(model, preview_image, "best_dice", i)
        

def convert_to_device(loaded: dict[str, Tensor]) -> list[Tensor]: return [x.to("AMD") for x in loaded.values()]

def load_dataset(filename: str) -> list[Tensor]: return convert_to_device(safe_load(filename))

if __name__ == "__main__":
  print(f"Loading train dataset from {TRAIN_DATASET}...")
  train = load_dataset(TRAIN_DATASET)
  print("Dataset loaded.")
  print(f"Loading validation dataset from {VAL_DATASET}...")
  val = load_dataset(VAL_DATASET)
  print("Dataset loaded.")
  run_training(train, val, argv[1] if len(argv) == 2 else None)
  
