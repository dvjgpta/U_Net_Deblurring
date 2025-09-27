import torch
from torch import nn
from data_loader import get_dataloaders
from data_augmentation import PairedTransforms
from preprocessing import *
from unet_model import UNet
from test_utils import test_model
from train_utils import *
from metrics_utils import lpips_fn

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Test dataloader
test_transforms = PairedTransforms(resize=(128,128), flip_prob=0.0)
_, test_loader = get_dataloaders(
    blurred_dir=train_dir_blurred,
    sharp_dir=train_dir_sharp,
    train_transforms=None,
    test_transforms=test_transforms,
    batch_size=64
)

# Model
model = UNet(n_channels=3, n_classes=3).to(DEVICE)
model.load_state_dict(torch.load("unet_model_v2.pth", map_location=DEVICE))

# Loss
# loss_fn = lambda y_pred, y_true: combined_loss(
#     y_pred, y_true,
#     alpha=1,  # weight for L1
#     beta=1,   # weight for SSIM
#     gamma=0.1,  # weight for LPIPS
#     lpips_fn=lpips_fn
# )

loss_fn = charbonnier_loss
# Evaluate
test_loss, test_psnr, test_ssim, test_lpips = test_model(model, test_loader, loss_fn, DEVICE)

print(f"Test Loss: {test_loss:.4f}")
print(f"PSNR: {test_psnr:.2f} dB")
print(f"SSIM: {test_ssim:.4f}")
print(f"LPIPS: {test_lpips:.4f}")
