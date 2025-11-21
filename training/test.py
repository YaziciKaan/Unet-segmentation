import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2
import numpy as np
from PIL import Image
import cv2
import os
import argparse
from tqdm import tqdm

from model import UNet


# Hyperparameters
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
IMAGE_HEIGHT = 160
IMAGE_WIDTH = 240


def load_model(checkpoint_path, device=DEVICE):
    """Load trained model from checkpoint."""
    model = UNet(in_channels=3, out_channels=1).to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    print(f"=> Model loaded from {checkpoint_path}")
    if "dice_score" in checkpoint:
        print(f"=> Model Dice Score: {checkpoint['dice_score']:.4f}")
    return model


def get_transform():
    """Get transformation for inference."""
    return A.Compose([
        A.Resize(height=IMAGE_HEIGHT, width=IMAGE_WIDTH),
        A.Normalize(
            mean=[0.0, 0.0, 0.0],
            std=[1.0, 1.0, 1.0],
            max_pixel_value=255.0,
        ),
        ToTensorV2(),
    ])


def predict_image(model, image_path, output_path, device=DEVICE, overlay=True):
    """
    Predict segmentation mask for a single image.
    
    Args:
        model: Trained model
        image_path: Path to input image
        output_path: Path to save output
        device: Device to run on
        overlay: Whether to overlay mask on original image
    """
    # Load and preprocess image
    original_image = Image.open(image_path).convert("RGB")
    original_size = original_image.size
    image = np.array(original_image)
    
    transform = get_transform()
    augmented = transform(image=image)
    image_tensor = augmented["image"].unsqueeze(0).to(device)
    
    # Predict
    with torch.no_grad():
        pred = torch.sigmoid(model(image_tensor))
        pred = (pred > 0.5).float()
    
    # Convert to numpy and resize to original size
    mask = pred.squeeze().cpu().numpy()
    mask = (mask * 255).astype(np.uint8)
    mask = cv2.resize(mask, original_size, interpolation=cv2.INTER_NEAREST)
    
    # Save results
    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else ".", exist_ok=True)
    
    if overlay:
        # Create overlay visualization
        original_np = np.array(original_image)
        colored_mask = np.zeros_like(original_np)
        colored_mask[:, :, 0] = mask  # Red channel for mask
        
        # Blend original image with mask
        alpha = 0.5
        overlay_img = cv2.addWeighted(original_np, 1-alpha, colored_mask, alpha, 0)
        
        Image.fromarray(overlay_img).save(output_path)
        print(f"=> Saved overlay result to {output_path}")
    else:
        Image.fromarray(mask).save(output_path)
        print(f"=> Saved mask to {output_path}")


def predict_images(model, input_dir, output_dir, device=DEVICE, overlay=True):
    """
    Predict segmentation masks for all images in a directory.
    
    Args:
        model: Trained model
        input_dir: Directory containing input images
        output_dir: Directory to save outputs
        device: Device to run on
        overlay: Whether to overlay mask on original images
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Get all image files
    image_files = [f for f in os.listdir(input_dir) 
                   if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))]
    
    print(f"Found {len(image_files)} images")
    
    for img_file in tqdm(image_files, desc="Processing images"):
        input_path = os.path.join(input_dir, img_file)
        output_path = os.path.join(output_dir, img_file)
        
        try:
            predict_image(model, input_path, output_path, device, overlay)
        except Exception as e:
            print(f"Error processing {img_file}: {str(e)}")


def predict_video(model, video_path, output_path, device=DEVICE, overlay=True):
    """
    Predict segmentation masks for video.
    
    Args:
        model: Trained model
        video_path: Path to input video
        output_path: Path to save output video
        device: Device to run on
        overlay: Whether to overlay mask on original video
    """
    # Open video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video {video_path}")
        return
    
    # Get video properties
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    transform = get_transform()
    
    print(f"Processing video: {total_frames} frames at {fps} FPS")
    
    for _ in tqdm(range(total_frames), desc="Processing video"):
        ret, frame = cap.read()
        if not ret:
            break
        
        # Convert BGR to RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Preprocess
        augmented = transform(image=frame_rgb)
        image_tensor = augmented["image"].unsqueeze(0).to(device)
        
        # Predict
        with torch.no_grad():
            pred = torch.sigmoid(model(image_tensor))
            pred = (pred > 0.5).float()
        
        # Convert to numpy and resize
        mask = pred.squeeze().cpu().numpy()
        mask = (mask * 255).astype(np.uint8)
        mask = cv2.resize(mask, (width, height), interpolation=cv2.INTER_NEAREST)
        
        if overlay:
            # Create visualization
            colored_mask = np.zeros_like(frame)
            colored_mask[:, :, 2] = mask  # Red channel (BGR format)
            
            # Blend
            alpha = 0.5
            overlay_frame = cv2.addWeighted(frame, 1-alpha, colored_mask, alpha, 0)
            out.write(overlay_frame)
        else:
            mask_bgr = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
            out.write(mask_bgr)
    
    cap.release()
    out.release()
    print(f"=> Saved output video to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="UNET Segmentation Inference")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/best.pth",
                        help="Path to model checkpoint")
    parser.add_argument("--mode", type=str, choices=["image", "images", "video"], required=True,
                        help="Inference mode: single image, multiple images, or video")
    parser.add_argument("--input", type=str, required=True,
                        help="Input image/video path or directory")
    parser.add_argument("--output", type=str, required=True,
                        help="Output path or directory")
    parser.add_argument("--no-overlay", action="store_true",
                        help="Save only mask without overlay")
    
    args = parser.parse_args()
    
    # Load model
    model = load_model(args.checkpoint, DEVICE)
    
    # Run inference based on mode
    overlay = not args.no_overlay
    
    if args.mode == "image":
        predict_image(model, args.input, args.output, DEVICE, overlay)
    elif args.mode == "images":
        predict_images(model, args.input, args.output, DEVICE, overlay)
    elif args.mode == "video":
        predict_video(model, args.input, args.output, DEVICE, overlay)
    
    print("\n✅ Inference completed!")


if __name__ == "__main__":
    main()
