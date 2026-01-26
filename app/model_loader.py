import os
import cv2
import time
import torch
import numpy as np
import albumentations as A

from app.model import UNet
from albumentations.pytorch import ToTensorV2

IMAGE_HEIGHT = 288
IMAGE_WIDTH = 512


try:
    import onnxruntime as ort
    ONNX_AVAILABLE = True
except ImportError:
    ONNX_AVAILABLE = False

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


class ModelWrapper:
    def __init__(self, model_path, device=DEVICE):
        self.model_path = model_path
        self.device = device
        self.model_type = self._detect_model_type()
        self.fps_history = []
        
        if self.model_type == "pytorch":
            self._load_pytorch_model()
        elif self.model_type == "onnx":
            self._load_onnx_model()
        else:
            raise ValueError(f"Unsupported model format: {model_path}")
    
    def _detect_model_type(self):
        ext = os.path.splitext(self.model_path)[1].lower()
        if ext in ['.pth', '.pt']:
            return "pytorch"
        elif ext == '.onnx':
            if not ONNX_AVAILABLE:
                raise ImportError("ONNX model detected but onnxruntime not installed!")
            return "onnx"
        else:
            return None
    
    def _load_pytorch_model(self):
        self.model = UNet(in_channels=3, out_channels=1).to(self.device)
        checkpoint = torch.load(self.model_path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(checkpoint["state_dict"])
        self.model.eval()
        
        self.transform = A.Compose([
            A.Resize(height=IMAGE_HEIGHT, width=IMAGE_WIDTH),
            A.Normalize(mean=[0.0, 0.0, 0.0], std=[1.0, 1.0, 1.0], max_pixel_value=255.0),
            ToTensorV2(),
        ])
        
        print(f"✓ PyTorch model loaded (Device: {self.device})")
        if "dice_score" in checkpoint:
            print(f"  Dice Score: {checkpoint['dice_score']:.4f}")
    
    def _load_onnx_model(self):
        providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
        self.session = ort.InferenceSession(self.model_path, providers=providers)
        self.input_name = self.session.get_inputs()[0].name
        actual_provider = self.session.get_providers()[0]
        print(f"✓ ONNX model loaded (Provider: {actual_provider})")
    
    def predict(self, frame):
        start_time = time.time()
        
        if self.model_type == "pytorch":
            mask = self._predict_pytorch(frame)
        else:
            mask = self._predict_onnx(frame)
        
        inference_time = time.time() - start_time
        fps = 1.0 / inference_time if inference_time > 0 else 0
        self.fps_history.append(fps)
        
        return mask, fps
    
    def _predict_pytorch(self, frame):
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        augmented = self.transform(image=frame_rgb)
        image_tensor = augmented["image"].unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            pred = torch.sigmoid(self.model(image_tensor))
            pred = (pred > 0.5).float()
        
        mask = pred.squeeze().cpu().numpy()
        mask = (mask * 255).astype(np.uint8)
        return mask
    
    def _predict_onnx(self, frame):
        frame_resized = cv2.resize(frame, (IMAGE_WIDTH, IMAGE_HEIGHT))
        frame_rgb = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)
        frame_normalized = frame_rgb.astype(np.float32) / 255.0
        frame_input = np.transpose(frame_normalized, (2, 0, 1))
        frame_input = np.expand_dims(frame_input, axis=0)
        
        outputs = self.session.run(None, {self.input_name: frame_input})
        mask_output = outputs[0]
        
        mask = 1 / (1 + np.exp(-mask_output))
        mask = (mask > 0.5).astype(np.float32)
        mask = mask.squeeze()
        mask = (mask * 255).astype(np.uint8)
        return mask
    
    def get_avg_fps(self):
        return np.mean(self.fps_history) if self.fps_history else 0