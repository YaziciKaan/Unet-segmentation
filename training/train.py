import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import albumentations as A
from albumentations.pytorch import ToTensorV2

from app.model import UNet
from dataset import SegmentationDataset
from app.utils import (
    save_checkpoint,
    check_accuracy,
)

# Hyperparameters
LEARNING_RATE = 1e-4
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 8
NUM_EPOCHS = 150
NUM_WORKERS = 2
IMAGE_HEIGHT = 288
IMAGE_WIDTH = 512
PIN_MEMORY = True
LOAD_MODEL = False
TRAIN_IMG_DIR = "/home/kaan/datasets/Pothole_Segmentation/train/images/"
TRAIN_MASK_DIR = "/home/kaan/datasets/Pothole_Segmentation/train/masks/"
VAL_IMG_DIR = "/home/kaan/datasets/Pothole_Segmentation/valid/images/"
VAL_MASK_DIR = "/home/kaan/datasets/Pothole_Segmentation/valid/masks/"


def train_fn(loader, model, optimizer, loss_fn, scaler):
    """
    Training function for one epoch.
    
    Args:
        loader: DataLoader for training data
        model: The neural network model
        optimizer: Optimizer for training
        loss_fn: Loss function
        scaler: GradScaler for mixed precision training
    """
    loop = tqdm(loader)
    
    for batch_idx, (data, targets) in enumerate(loop):
        data = data.to(device=DEVICE)
        targets = targets.float().unsqueeze(1).to(device=DEVICE)
        
        # Forward pass with mixed precision
        with torch.amp.autocast('cuda'):
            predictions = model(data)
            loss = loss_fn(predictions, targets)
        
        # Backward pass
        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        
        # Update progress bar
        loop.set_postfix(loss=loss.item())


def main():
    # Define transforms for training
    train_transform = A.Compose([
        A.RandomCrop(height=IMAGE_HEIGHT, width=IMAGE_WIDTH),
        A.Rotate(limit=35, p=1.0),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.1),
        A.Normalize(
            mean=[0.0, 0.0, 0.0],
            std=[1.0, 1.0, 1.0],
            max_pixel_value=255.0,
        ),
        ToTensorV2(),
    ])
    
    # Define transforms for validation
    val_transform = A.Compose([
        A.Resize(height=IMAGE_HEIGHT, width=IMAGE_WIDTH),
        A.Normalize(
            mean=[0.0, 0.0, 0.0],
            std=[1.0, 1.0, 1.0],
            max_pixel_value=255.0,
        ),
        ToTensorV2(),
    ])
    
    # Initialize model, loss function, and optimizer
    model = UNet(in_channels=3, out_channels=1).to(DEVICE)
    loss_fn = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    # Create datasets
    train_dataset = SegmentationDataset(
        image_dir=TRAIN_IMG_DIR,
        mask_dir=TRAIN_MASK_DIR,
        transform=train_transform,
    )
    
    val_dataset = SegmentationDataset(
        image_dir=VAL_IMG_DIR,
        mask_dir=VAL_MASK_DIR,
        transform=val_transform,
    )
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY,
        shuffle=True,
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY,
        shuffle=False,
    )
    
    check_accuracy(val_loader, model, device=DEVICE)
    
    scaler = torch.amp.GradScaler('cuda')
    best_dice = 0.0
    
    # Training loop
    for epoch in range(NUM_EPOCHS):
        print(f"\nEpoch {epoch+1}/{NUM_EPOCHS}")
        train_fn(train_loader, model, optimizer, loss_fn, scaler)
        
        # Check accuracy on validation set
        accuracy, dice_score = check_accuracy(val_loader, model, device=DEVICE)
        
        # Check if this is the best model
        is_best = dice_score > best_dice
        if is_best:
            best_dice = dice_score
        
        # Save checkpoint with epoch and metrics info
        checkpoint = {
            "epoch": epoch + 1,
            "state_dict": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "dice_score": dice_score,
            "accuracy": accuracy,
            "best_dice": best_dice,
        }
        
        # Save latest and best every epoch, periodic checkpoint every 10 epochs
        save_periodic = (epoch + 1) % 10 == 0
        save_checkpoint(checkpoint, is_best=is_best, save_periodic=save_periodic)


if __name__ == "__main__":
    main()
