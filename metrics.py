# # metrics.py
# import torch
# import matplotlib.pyplot as plt
# from helper_functions import *
# from data_augmentation import * 
# from unet_model import *
# from data_loader import *
# from preprocessing import *

# # -------------------------
# # Config
# # -------------------------

# results = torch.load("/mnt/DATA/EE22B013/Btech_project/U_NET/results.pth")
# DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# MODEL_PATH = "/mnt/DATA/EE22B013/Btech_project/U_NET/unet_model.pth"   # Path where trained model is saved
# BATCH_SIZE = 64
# NUM_IMAGES = 5

# # -------------------------
# # Load trained model
# # -------------------------
# model_0 = UNet(n_channels=3, n_classes=3).to(DEVICE)
# model_0.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))

# model_0.eval()

# # -------------------------
# # Load training results
# # -------------------------

# # -------------------------
# # Get dataloaders
# # -------------------------
# train_dataloader, test_dataloader = get_dataloaders(
#     blurred_dir=train_dir_blurred,
#     sharp_dir=train_dir_sharp,
#     train_transforms=train_transforms,
#     test_transforms=test_transforms,
#     batch_size=BATCH_SIZE
# )

# # -------------------------
# # Utility functions
# # -------------------------
# def count_parameters(model):
#     total_params = sum(p.numel() for p in model.parameters())
#     trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
#     print(f"Total parameters: {total_params}")
#     print(f"Trainable parameters: {trainable_params}")

# def plot_training_metrics(results):
#     epochs = range(1, len(results['train_loss']) + 1)
#     fig, axes = plt.subplots(1, 2, figsize=(12, 5))

#     # Loss
#     axes[0].plot(epochs, results['train_loss'], label='Train Loss', marker='o')
#     axes[0].plot(epochs, results['test_loss'], label='Test Loss', marker='o')
#     axes[0].set_title("Loss over Epochs")
#     axes[0].set_xlabel("Epoch")
#     axes[0].set_ylabel("Loss")
#     axes[0].legend()
#     axes[0].grid(True)

#     # PSNR
#     axes[1].plot(epochs, results['train_psnr'], label='Train PSNR', marker='o')
#     axes[1].plot(epochs, results['test_psnr'], label='Test PSNR', marker='o')
#     axes[1].set_title("PSNR over Epochs")
#     axes[1].set_xlabel("Epoch")
#     axes[1].set_ylabel("PSNR (dB)")
#     axes[1].legend()
#     axes[1].grid(True)

#     plt.tight_layout()
#     plt.show()


# def denormalize(tensor):
#     """
#     Convert model output from [-1,1] to [0,1] for plotting.
#     """
#     tensor = (tensor + 1) / 2
#     tensor = torch.clamp(tensor, 0, 1)
#     return tensor

# def visualize_predictions(model, dataloader, device='cuda', num_images=5, normalize_range='0_to_1'):
#     """
#     Display a few test images: blurred input, ground truth, predicted output.
    
#     Args:
#         model: trained PyTorch model
#         dataloader: DataLoader for test/validation set
#         device: 'cuda' or 'cpu'
#         num_images: number of images to display
#         normalize_range: range of your dataset; '-1_to_1' or '0_to_1'
#     """
#     model.eval()
#     device = torch.device(device if torch.cuda.is_available() else 'cpu')

#     # Get one batch
#     X, y = next(iter(dataloader))
#     X, y = X.to(device), y.to(device)

#     with torch.inference_mode():
#         preds = model(X)

#     # Move to CPU and permute to [B,H,W,C] for matplotlib
#     X = X.cpu().permute(0, 2, 3, 1)
#     y = y.cpu().permute(0, 2, 3, 1)
#     preds = preds.cpu().permute(0, 2, 3, 1)

#     # -------------------------------
#     # Denormalize / Clamp
#     # -------------------------------
#     if normalize_range == '-1_to_1':
#         X = torch.clamp((X + 1) / 2, 0, 1)
#         y = torch.clamp((y + 1) / 2, 0, 1)
#         preds = torch.clamp((preds + 1) / 2, 0, 1)
#     elif normalize_range == '0_to_1':
#         X = torch.clamp(X, 0, 1)
#         y = torch.clamp(y, 0, 1)
#         preds = torch.clamp(preds, 0, 1)
#     else:
#         raise ValueError("normalize_range must be '-1_to_1' or '0_to_1'")

#     # -------------------------------
#     # Debug: print min/max
#     # -------------------------------
#     print(f"Input: min={X.min():.4f}, max={X.max():.4f}")
#     print(f"Ground truth: min={y.min():.4f}, max={y.max():.4f}")
#     print(f"Prediction: min={preds.min():.4f}, max={preds.max():.4f}")

#     # -------------------------------
#     # Plot images
#     # -------------------------------
#     fig, axes = plt.subplots(3, num_images, figsize=(15, 5))
#     for i in range(num_images):
#         axes[0, i].imshow(X[i])
#         axes[0, i].set_title("Blurred")
#         axes[0, i].axis("off")

#         axes[1, i].imshow(y[i])
#         axes[1, i].set_title("Sharp GT")
#         axes[1, i].axis("off")

#         axes[2, i].imshow(preds[i])
#         axes[2, i].set_title("Predicted")
#         axes[2, i].axis("off")

#     plt.tight_layout()
#     plt.show()


# # -------------------------
# # Run metrics
# # -------------------------
# if __name__ == "__main__":
#     count_parameters(model_0)
#     plot_training_metrics(results)
#     visualize_predictions(model_0, test_dataloader, device=DEVICE, num_images=NUM_IMAGES)
# metrics.py
import torch
from unet_model import UNet
from data_loader import get_dataloaders
from data_augmentation import PairedTransforms
from metrics_utils import count_parameters, plot_training_metrics
from visualisation_utils import visualize_predictions
from preprocessing import *

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
NUM_IMAGES = 5

# Load trained model and results
model = UNet(n_channels=3, n_classes=3).to(DEVICE)
model.load_state_dict(torch.load("unet_model_v2.pth", map_location=DEVICE))
results = torch.load("results_v2.pth")

count_parameters(model)
plot_training_metrics(results)

# Test dataloader
test_transforms = PairedTransforms(resize=(128,128), flip_prob=0.0)
_, test_loader = get_dataloaders(
    blurred_dir=train_dir_blurred,
    sharp_dir=train_dir_sharp,
    train_transforms=None,
    test_transforms=test_transforms,
    batch_size=64
)

visualize_predictions(model, test_loader, device=DEVICE, num_images=NUM_IMAGES)
