import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2
import numpy as np
from PIL import Image
import cv2
import os
import argparse
import time

from app.model import UNet
from train import IMAGE_HEIGHT, IMAGE_WIDTH

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


def predict_video_realtime(model_wrapper, video_path, output_path=None, show_gui=True):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video {video_path}")
        return
    
    fps_orig = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    print(f"\nVideo: {width}x{height} @ {fps_orig} FPS, {total_frames} frames")
    print(f"Model: {model_wrapper.model_type.upper()}")
    
    gui_available = show_gui
    if show_gui:
        try:
            test_window = 'opencv_test_window'
            cv2.namedWindow(test_window, cv2.WINDOW_NORMAL)
            cv2.destroyWindow(test_window)
            print("GUI mode - Press 'q' to quit, 's' to screenshot\n")
        except (cv2.error, Exception):
            gui_available = False
            print("⚠️  GUI not available (headless/SSH mode). Running without display.\n")
    
    out = None
    if output_path:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps_orig, (width, height))
    
    frame_count = 0
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_count += 1
            
            # Predict
            mask, fps = model_wrapper.predict(frame)
            
            # Resize mask to original size
            mask_resized = cv2.resize(mask, (width, height), interpolation=cv2.INTER_NEAREST)
            
            # Create overlay
            colored_mask = np.zeros_like(frame)
            colored_mask[:, :, 2] = mask_resized  # Red channel (BGR)
            alpha = 0.5
            overlay_frame = cv2.addWeighted(frame, 1-alpha, colored_mask, alpha, 0)
            
            # Add FPS
            cv2.putText(overlay_frame, f"FPS: {fps:.1f} | Frame: {frame_count}/{total_frames}", 
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            cv2.putText(overlay_frame, f"Model: {model_wrapper.model_type.upper()}", 
                       (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            
            # Save to video
            if out:
                out.write(overlay_frame)
            
            # Show GUI
            if gui_available:
                try:
                    cv2.imshow('Segmentation - Press q to quit', overlay_frame)
                    key = cv2.waitKey(1) & 0xFF
                    if key == ord('q'):
                        break
                    elif key == ord('s'):
                        screenshot_path = f"screenshot_{frame_count}.jpg"
                        cv2.imwrite(screenshot_path, overlay_frame)
                        print(f"✓ Screenshot saved: {screenshot_path}")
                except (cv2.error, Exception):
                    gui_available = False
                    print("\n⚠️  GUI error - switching to headless mode")
            
            if not gui_available and frame_count % 50 == 0:
                print(f"Processed {frame_count}/{total_frames} frames - {fps:.1f} FPS")
    
    except KeyboardInterrupt:
        print("\n\nStopped by user")
    
    finally:
        cap.release()
        if out:
            out.release()
        if gui_available:
            try:
                cv2.destroyAllWindows()
            except (cv2.error, Exception):
                pass
        
        print(f"\n{'='*60}")
        print(f"Processed {frame_count} frames")
        print(f"Average FPS: {model_wrapper.get_avg_fps():.2f}")
        print(f"Average Inference Time: {1000/model_wrapper.get_avg_fps():.2f} ms")
        print('='*60)
        
        if output_path:
            print(f"\n✓ Output saved: {output_path}")


def predict_image(model_wrapper, image_path, output_path, overlay=True):
    original_image = cv2.imread(image_path)
    if original_image is None:
        print(f"Error: Could not load image {image_path}")
        return
    
    mask, fps = model_wrapper.predict(original_image)
    
    h, w = original_image.shape[:2]
    mask_resized = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)
    
    if overlay:
        colored_mask = np.zeros_like(original_image)
        colored_mask[:, :, 2] = mask_resized
        alpha = 0.5
        result = cv2.addWeighted(original_image, 1-alpha, colored_mask, alpha, 0)
    else:
        result = cv2.cvtColor(mask_resized, cv2.COLOR_GRAY2BGR)
    
    os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
    cv2.imwrite(output_path, result)
    print(f"✓ Saved: {output_path} (FPS: {fps:.1f})")


def main():
    parser = argparse.ArgumentParser(description="UNET Segmentation Inference (PyTorch or ONNX)")
    parser.add_argument("--model", type=str, default=None,
                        help="Path to model (.pth for PyTorch, .onnx for ONNX). Default: ../pothole_unet.onnx")
    parser.add_argument("--mode", type=str, choices=["image", "video"], required=True,
                        help="Inference mode")
    parser.add_argument("--input", type=str, required=True,
                        help="Input image or video path")
    parser.add_argument("--output", type=str, default=None,
                        help="Output path (optional)")
    parser.add_argument("--no-gui", action="store_true",
                        help="Disable GUI for video mode")
    parser.add_argument("--no-overlay", action="store_true",
                        help="Save only mask without overlay (image mode)")
    
    args = parser.parse_args()
    
    # Default to ONNX model if not specified
    if args.model is None:
        args.model = os.path.join(os.path.dirname(os.path.dirname(__file__)), "pothole_unet.onnx")
        print(f"No model specified, using default: {args.model}")
    
    # Load model
    print(f"\nLoading model: {args.model}")
    print("="*60)
    model_wrapper = ModelWrapper(args.model)
    print("="*60)
    
    # Run inference
    if args.mode == "image":
        output = args.output or "output.jpg"
        predict_image(model_wrapper, args.input, output, not args.no_overlay)
    elif args.mode == "video":
        show_gui = not args.no_gui
        predict_video_realtime(model_wrapper, args.input, args.output, show_gui)


if __name__ == "__main__":
    main()
