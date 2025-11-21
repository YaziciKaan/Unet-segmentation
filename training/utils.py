import torch
import os


def save_checkpoint(state, is_best=False, save_periodic=False, checkpoint_dir="checkpoints"):
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    # Save latest model
    latest_path = os.path.join(checkpoint_dir, "latest.pth")
    torch.save(state, latest_path)
    print(f"=> Saved latest checkpoint (Epoch {state['epoch']})")
    
    # Save best model
    if is_best:
        best_path = os.path.join(checkpoint_dir, "best.pth")
        torch.save(state, best_path)
        print(f"=> Saved as BEST model! (Dice: {state['dice_score']:.4f})")
    
    # Save periodic checkpoint (every N epochs)
    if save_periodic:
        periodic_path = os.path.join(checkpoint_dir, f"checkpoint_epoch_{state['epoch']}.pth")
        torch.save(state, periodic_path)
        print(f"=> Saved periodic checkpoint")


def check_accuracy(loader, model, device="cuda"):
    num_correct = 0
    num_pixels = 0
    dice_score = 0
    model.eval()

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            y = y.to(device).unsqueeze(1)
            preds = torch.sigmoid(model(x))
            preds = (preds > 0.5).float()
            num_correct += (preds == y).sum()
            num_pixels += torch.numel(preds)
            dice_score += (2 * (preds * y).sum()) / (
                (preds + y).sum() + 1e-8
            )

    accuracy = num_correct/num_pixels*100
    dice = dice_score/len(loader)
    
    print(
        f"Got {num_correct}/{num_pixels} with acc {accuracy:.2f}%"
    )
    print(f"Dice score: {dice:.4f}")
    model.train()
    
    return accuracy.item(), dice.item()