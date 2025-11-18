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
from inference import infer_and_overlap


DATASET = "dataset/MICCAI_BraTS_2019_Data_Training/*GG/*"
TEST_IMAGE = "dataset/MICCAI_BraTS_2019_Data_Training/LGG/BraTS19_2013_0_1"
EPOCHS = 500


def validate(model: UNet, dataset: Dataset) -> float:
    """Validate the model by the means of calculating the DICE coefficient"""
    total_dice = 0.0
    l = len(dataset.images) 
    
    @TinyJit
    def f(idx: int) -> Tensor:
        pred = model(dataset.images[idx])
        target = dataset.labels[idx]
        smooth = 1e-5
        y = (pred.sigmoid() > 0.5).float()
        return (2.0 * (y * target).sum() + smooth) / (y.sum() + target.sum() + smooth)

      
    with Tensor.train(False):
      for idx in tqdm(range(l), desc="Validating"): total_dice += f(idx).numpy()
      
    return total_dice / l


@TinyJit
def tiny_step(idx: int, dataset: Tensor, model: UNet, optimizer: Adam) -> Tensor:
    optimizer.zero_grad()
    pred = model(dataset.images[idx]).sigmoid()
    label = dataset.labels[idx]
    # loss = pred.binary_crossentropy_logits(label)
    smooth = 1e-5
    y = (pred.sigmoid() > 0.5).float()
    dice_loss = 1.0 - (2.0 * (pred * label).sum() + smooth) / (pred.sum() + label.sum() + smooth)
    loss = 0.8 * dice_loss + 0.2 * pred.binary_crossentropy_logits(label)
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
        # TODO: Maybe print the gradients here to check whether they are diminishing.
        total_loss += loss.numpy()
    
    return total_loss / l
    

def run_training(train: Tensor, val: Tensor):
  model = UNet()
  optim = Adam(get_parameters(model), 1e-3)
  dice = 0
  largest_dice = 0
  for i in range(1, EPOCHS+1):
    epoch_msg = f"\nEpoch {i}/{EPOCHS}"
    print(epoch_msg)
    train_loss = train_epoch(model, train, optim)
    print(f"Train Loss: {train_loss:.6f}")
    dice = validate(model, val)
    val_msg = ", ".join([epoch_msg, f"Epoch Validation Dice: {dice:.6f}"])
    print(val_msg)
    with open("eval_scores.txt", "a") as f: f.write(val_msg)
    if i % 10 == 0:
        with Tensor.train(False): infer_and_overlap(model, TEST_IMAGE, "training_validation", i)
    if dice > largest_dice:
        print("This is the best DICE so far!!!")
        largest_dice = dice
        with Tensor.train(False): infer_and_overlap(model, TEST_IMAGE, "best_dice", i)
        safe_save(get_state_dict(model), f"model_dice_{dice}_percent.safetensors")


if __name__ == "__main__":
  train, val = [Dataset(x) for x in choose_files(DATASET)]
  run_training(train, val)
  
