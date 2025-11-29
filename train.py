#!/usr/bin/env python3

from net import UNet
from tinygrad.tensor import Tensor
from tinygrad.engine.jit import TinyJit
from tinygrad.nn.optim import AdamW
from tinygrad.nn.state import get_parameters, load_state_dict, safe_load
from tinygrad.helpers import tqdm
from tinygrad.nn.state import safe_save, get_state_dict
from dataset import REGEX, choose_files, Dataset, load_mask
from random import shuffle
from inference import infer_and_overlap
from sys import argv
import json


EPOCHS = 500


def validate(model: UNet, dataset: Dataset) -> float:
    """Validate the model by the means of calculating the DICE coefficient"""
    total_dice = 0.0
    l = len(dataset.images) 
    
    @TinyJit
    def f(idx: int) -> Tensor:
        pred = model(dataset.images[idx])
        probs = pred.sigmoid()
        label = dataset.labels[idx]
        smooth = 1e-6
        return (2.0 * (probs * label).sum() + smooth) / (probs.sum() + label.sum() + smooth)
      
    with Tensor.train(False):
      for idx in tqdm(range(l), desc="Validating"): total_dice += f(idx).numpy()
      
    return total_dice / l


@TinyJit
def tiny_step(idx: int, dataset: Tensor, model: UNet, optimizer: AdamW) -> Tensor:
    optimizer.zero_grad()
    logits = model(dataset.images[idx])
    label = dataset.labels[idx]
    smooth = 1e-6

    # Focal loss
    gamma = 2.0
    alpha = 0.25 # TODO: For alpha-balanced CE. Try to set this by inverse class frequency (like explained in the Focal loss paper).
    bce = logits.binary_crossentropy_logits(label, reduction="none")
    pt = (-bce).exp()
    focal_weight = (label * alpha + (1.0 - label) * (1.0 - alpha)) * (1.0 - pt).pow(gamma)
    focal_loss = (focal_weight * bce).mean()

    # Tversky loss
    probs = logits.sigmoid()
    alpha_t = 0.3
    beta_t = 0.7
    gamma_t = 4/3
    tp = (probs * label).sum()
    fp = ((1.0 - label) * probs).sum()
    fn = (label * (1.0 - probs)).sum()
    focal_tversky_loss = (1.0 - (tp + smooth) / (tp + alpha_t * fp + beta_t * fn + smooth)).pow(gamma_t)

    loss = 0.4 * focal_loss + 0.6 * focal_tversky_loss
    loss.backward()
    optimizer.step()
    return loss


def train_epoch(model: UNet, dataset: Dataset, optimizer: AdamW) -> float:
    """Train for one epoch"""
    total_loss = 0.0
    l = len(dataset.images)
    
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
    for example in examples:
        if (load_mask(example).max() > 0).numpy(): return example
    return None
    

def run_training(train: Tensor, val: Tensor):
  model = UNet()
  if len(argv) > 1:
    state = safe_load(argv[1])  # Load model checkpoint
    load_state_dict(model, state)
    print(f"Loaded model from {argv[1]}.")
  optim = AdamW(get_parameters(model), 1e-3)
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
    print(f"Train Loss: {train_loss:.6f}")
    dice = validate(model, val)
    val_msg = ", ".join([epoch_msg, f"Epoch Validation Dice: {dice:.6f}"])
    print(val_msg)
    with open("eval_scores.txt", "a") as f: f.write(val_msg)
    if i % 10 == 0:
        with Tensor.train(False): infer_and_overlap(model, preview_image, "training_validation", i)
    if i % 50 == 0:
        print("Saving the model...")
        safe_save(get_state_dict(model), f"model_epoch_{i}.safetensors")
        print("Saved.")
    if dice > largest_dice:
        print("This is the best DICE so far!!!")
        largest_dice = dice
        with Tensor.train(False): infer_and_overlap(model, preview_image, "best_dice", i)


if __name__ == "__main__":
  print("Loading dataset...")
  train, val = [Dataset(x) for x in choose_files(REGEX)]
  print("Dataset loaded.")
  run_training(train, val)
  