#!/usr/bin/env python3

from net import UNet
from tinygrad.tensor import Tensor
from tinygrad.engine.jit import TinyJit
from tinygrad.nn.optim import Adam
from tinygrad.nn.state import get_parameters
from tinygrad.helpers import tqdm
from tinygrad.nn.state import safe_save, get_state_dict
from dataset import choose_files, Dataset
from random import shuffle


DATASET = "dataset/MICCAI_BraTS_2019_Data_Training/*GG/*"
EPOCHS = 50


def dice_coefficient(pred: Tensor, target: Tensor, threshold=0.5, smooth=1e-5) -> Tensor:
    """Dice coefficient metric"""
    pred_binary = (pred > threshold).float()
    intersection = (pred_binary * target).sum()
    union = pred_binary.sum() + target.sum()
    dice = (2.0 * intersection + smooth) / (union + smooth)
    return dice


def validate(model: UNet, dataset: Dataset) -> float:
    """Validate the model"""
    total_dice = 0.0
    l = len(dataset.images) 
    
    @TinyJit
    def f(idx: int) -> Tensor:
      image = dataset.images[idx]
      pred = model(image)
      return dice_coefficient(pred, dataset.labels[idx])
      
    with Tensor.train(False):
      for idx in tqdm(range(l), desc="Validating"): total_dice += f(idx).numpy()
      
    return total_dice / l


@TinyJit
def tiny_step(idx: int, dataset: Tensor, model: UNet, optimizer: Adam) -> Tensor:
    optimizer.zero_grad()
    pred = model(dataset.images[idx])
    label = dataset.labels[idx]
    loss = pred.binary_crossentropy_logits(label)
    loss.backward()
    optimizer.step()
    return loss


def train_epoch(model: UNet, dataset: Dataset, optimizer: Adam) -> float:
    """Train for one epoch"""
    total_loss = 0.0
    l = len(dataset.images)
    
    indices = list(range(l))
    shuffle(indices)
    
    with Tensor.train(True):
      for idx in tqdm(indices, desc="Training"):
        loss = tiny_step(idx, dataset, model, optimizer)
        total_loss += loss.numpy()
    
    return total_loss / l
    

def run_training(train: Tensor, val: Tensor):
  model = UNet()
  optim = Adam(get_parameters(model), 1e-4)
  for i in range(EPOCHS):
    epoch_msg = f"\nEpoch {i+1}/{EPOCHS}"
    print(epoch_msg)
    # TODO: Loss function returns nan. Something is bad with the prediction or the mask probably
    train_loss = train_epoch(model, train, optim)
    print(f"Train Loss: {train_loss:.4f}")
    val_dice = validate(model, val)
    val_msg = ", ".join([epoch_msg, f"Epoch Validation Dice: {val_dice:.4f}"])
    print(val_msg)
    with open("eval_scores.txt", "a") as f: f.write(val_msg)
  safe_save(get_state_dict(model), f"model{EPOCHS}.safetensors")


if __name__ == "__main__":
  train, val = [Dataset(x) for x in choose_files(DATASET)]
  run_training(train, val)
  
