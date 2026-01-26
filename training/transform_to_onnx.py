import torch
from model import UNet

IMAGE_HEIGHT = 288
IMAGE_WIDTH = 512

model = UNet(in_channels=3, out_channels=1)
model.load_state_dict(torch.load('./models/best_model.pth')["state_dict"])
model.eval()

dummy_input = torch.randn(1, 3, IMAGE_HEIGHT, IMAGE_WIDTH)
torch.onnx.export(model, dummy_input, "best_model.onnx")