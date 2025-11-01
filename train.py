#!/usr/bin/env python3

from net import UNet
from tinygrad.tensor import Tensor
from tinygrad.engine.jit import TinyJit
from tinygrad.nn.optim import Adam
from tinygrad.nn.state import get_parameters
from tinygrad.helpers import tqdm
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
def tiny_step(idx: int, dataset: Tensor, model: UNet, optimizer: Adam) -> tuple[Tensor, Tensor]:
    pred = model(dataset.images[idx])
    label = dataset.labels[idx]
    loss = pred.binary_crossentropy(label)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    dice = dice_coefficient(pred, label)
    return loss, dice


def train_epoch(model: UNet, dataset: Dataset, optimizer: Adam) -> tuple[float, float]:
    """Train for one epoch"""
    total_loss = 0.0
    total_dice = 0.0
    l = len(dataset.images)
    
    indices = list(range(l))
    shuffle(indices)
    
    with Tensor.train(True):
      for idx in tqdm(indices, desc="Training"):
        loss, dice = tiny_step(idx, dataset, model, optimizer)
        total_loss += loss.numpy()
        total_dice += dice.numpy()
    
    avg_loss = total_loss / l
    avg_dice = total_dice / l
    
    return avg_loss, avg_dice
    

def run_training(train: Tensor, val: Tensor):
  model = UNet()
  optim = Adam(get_parameters(model), 1e-4)
  for i in range(EPOCHS):
    epoch_msg = f"\nEpoch {i+1}/{EPOCHS}"
    print(epoch_msg)
    train_loss, train_dice = train_epoch(model, train, optim)
    print(f"Train Loss: {train_loss:.4f}; Train DICE: {train_dice:.4f}")
    val_dice = validate(model, val)
    print(", ".join([epoch_msg, f"Epoch Validation Dice: {val_dice:.4f}"]))


if __name__ == "__main__":
  train, val = [Dataset(x) for x in choose_files(DATASET)]
  run_training(train, val)
  