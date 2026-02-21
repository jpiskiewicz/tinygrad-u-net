#!/usr/bin/env python3

from tinygrad.tensor import Tensor
from tinygrad.engine.jit import TinyJit
from tinygrad.nn.optim import Adam, Optimizer
from tinygrad.nn.state import get_parameters, load_state_dict, safe_load, safe_save, get_state_dict
from tinygrad.helpers import tqdm
from tinygrad_unet.net import UNet
from tinygrad_unet.dataset import TRAIN_DATASET, VAL_DATASET, load_dataset
from tinygrad_unet.inference import load_combined_mask, infer_and_overlap
from datetime import datetime
from random import shuffle
from pathlib import Path
from sys import argv
import json
import glob


EPOCHS = 500
SAVE_BEST_DICE_PREDICTION = False


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


def train_epoch(model: UNet, dataset: list[Tensor], optimizer: Optimizer) -> float:
    """Train for one epoch"""
    total_loss = 0.0
    l = int(dataset[0].shape[0])
    indices = list(range(l))
    shuffle(indices)

    with Tensor.train(True):
      for i in tqdm(indices, desc="Training"):
        example, label = dataset[0][i].contiguous(), dataset[1][i].contiguous()
        loss = tiny_step(example, label, model, optimizer)
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
    

def load_augumented_dataset(i: int) -> list[Tensor]:
  print(f"{TRAIN_DATASET.split('.')[0]}*.safetensors") # TODO: This glob patterns is somehow fucked
  training_sets = glob.glob(f"../{TRAIN_DATASET.split('.')[0]}*.safetensor")
  print(sorted(training_sets))
    

def run_training(val: list[Tensor], epochs: int, model_file: str | None):
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
  for i in range(1 if model_file is None else int(model_file.split("_")[2][:-12]), epochs+1):
    epoch_msg = f"\nEpoch {i}/{epochs}"
    print(epoch_msg)
    train = load_augumented_dataset(i)
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
        if SAVE_BEST_DICE_PREDICTION:
            with Tensor.train(False): infer_and_overlap(model, preview_image, "best_dice", i)
        

if __name__ == "__main__":
  load_augumented_dataset(0)
  print(f"Loading validation dataset from {VAL_DATASET}...")
  val = load_dataset(VAL_DATASET)
  print("Dataset loaded.")
  run_training(val, int(argv[1]) if len(argv) == 2 else EPOCHS, argv[2] if len(argv) == 3 else None)
