import os
import numpy as np
from PIL import Image, ImageDraw
from tqdm import tqdm


def yolo_to_mask(txt_path, img_width, img_height):
    """
    Convert YOLOv8 segmentation format (.txt) to binary mask.
    
    Args:
        txt_path (str): Path to .txt file with normalized coordinates
        img_width (int): Image width
        img_height (int): Image height
        
    Returns:
        PIL.Image: Binary mask
    """
    mask = Image.new('L', (img_width, img_height), 0)
    
    if not os.path.exists(txt_path):
        return mask
    
    with open(txt_path, 'r') as f:
        lines = f.readlines()
    
    draw = ImageDraw.Draw(mask)
    
    for line in lines:
        coords = line.strip().split()
        if len(coords) < 3:
            continue
            
        class_id = int(coords[0])
        
        points = []
        for i in range(1, len(coords), 2):
            if i + 1 < len(coords):
                x = float(coords[i]) * img_width
                y = float(coords[i + 1]) * img_height
                points.append((x, y))
        
        if len(points) >= 3:
            draw.polygon(points, outline=255, fill=255)
    
    return mask


def convert_dataset(image_dir, label_dir, output_mask_dir):
    # Create output directory if it doesn't exist
    os.makedirs(output_mask_dir, exist_ok=True)
    
    image_files = [f for f in os.listdir(image_dir) 
                   if f.endswith(('.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG'))]
    
    print(f"Found {len(image_files)} images")
    print(f"Converting YOLO labels to masks...")
    
    successful = 0
    failed = 0
    
    for img_name in tqdm(image_files):
        try:
            img_path = os.path.join(image_dir, img_name)
            txt_name = os.path.splitext(img_name)[0] + '.txt'
            txt_path = os.path.join(label_dir, txt_name)
            
            image = Image.open(img_path)
            img_width, img_height = image.size
            
            # Convert YOLO format to mask
            mask = yolo_to_mask(txt_path, img_width, img_height)
            
            # Save mask with same name as image
            mask_name = os.path.splitext(img_name)[0] + '.png'
            mask_path = os.path.join(output_mask_dir, mask_name)
            mask.save(mask_path)
            
            successful += 1
            
        except Exception as e:
            print(f"\nError processing {img_name}: {str(e)}")
            failed += 1
    
    print(f"\nConversion complete!")
    print(f"Successfully converted: {successful}")
    print(f"Failed: {failed}")
    print(f"Masks saved to: {output_mask_dir}")


if __name__ == "__main__":
    print("=" * 50)
    print("Converting TRAIN set")
    print("=" * 50)
    convert_dataset(
        image_dir="/home/kaan/datasets/Pothole_Segmentation/train/images/",
        label_dir="/home/kaan/datasets/Pothole_Segmentation/train/labels/",
        output_mask_dir="/home/kaan/datasets/Pothole_Segmentation/train/masks/"
    )
    
    print("\n" + "=" * 50)
    print("Converting VALIDATION set")
    print("=" * 50)
    convert_dataset(
        image_dir="/home/kaan/datasets/Pothole_Segmentation/valid/images/",
        label_dir="/home/kaan/datasets/Pothole_Segmentation/valid/labels/",
        output_mask_dir="/home/kaan/datasets/Pothole_Segmentation/valid/masks/"
    )
    
    print("\n✅ All done! You can now use the masks for training.")
