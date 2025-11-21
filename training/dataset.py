import os
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
import torch


class SegmentationDataset(Dataset):
    def __init__(self, image_dir, mask_dir, transform=None):
        """
        Dataset class for image segmentation tasks.
        
        Args:
            image_dir (str): Directory containing input images
            mask_dir (str): Directory containing segmentation masks
            transform (callable, optional): Optional transform to be applied on both image and mask
        """
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.transform = transform
        self.images = [f for f in os.listdir(image_dir) 
                      if f.endswith(('.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG'))]

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        img_name = self.images[index]
        img_path = os.path.join(self.image_dir, img_name)
        
        # Get corresponding mask (same name but .png extension)
        mask_name = os.path.splitext(img_name)[0] + '.png'
        mask_path = os.path.join(self.mask_dir, mask_name)
        
        # Load image and mask
        image = np.array(Image.open(img_path).convert("RGB"))
        mask = np.array(Image.open(mask_path).convert("L"), dtype=np.float32)
        
        # Normalize mask to 0-1 range if needed
        if mask.max() > 1.0:
            mask = mask / 255.0

        # Apply transformations if provided
        if self.transform is not None:
            augmentations = self.transform(image=image, mask=mask)
            image = augmentations["image"]
            mask = augmentations["mask"]

        return image, mask
