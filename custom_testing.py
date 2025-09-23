# custom_testing.py
import torch
from torch import nn
from unet_model import UNet
from visualisation_utils import *

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = UNet(n_channels=3, n_classes=3).to(DEVICE)
model.load_state_dict(torch.load("unet_model.pth", map_location=DEVICE))

img_path = "/mnt/DATA/EE22B013/Btech_project/U_NET/blurred_1.jpg"  # Replace with your image path

out_img = custom_test(model, img_path, device="cuda", save_path="deblurred.jpg")
out_img.show()
