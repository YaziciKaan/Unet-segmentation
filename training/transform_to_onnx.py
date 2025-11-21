import torch
from model import UNet
from train import IMAGE_HEIGHT, IMAGE_WIDTH

model = UNet(in_channels=3, out_channels=1)
model.load_state_dict(torch.load('checkpoints/best.pth')["state_dict"])
model.eval()

dummy_input = torch.randn(1, 3, IMAGE_HEIGHT, IMAGE_WIDTH)
torch.onnx.export(model, dummy_input, "pothole_unet.onnx")