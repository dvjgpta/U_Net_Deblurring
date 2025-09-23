# import torch
# import torchinfo
# import torch.nn.functional as F
# import time
# from torchinfo import summary
# from preprocessing import *
# from data_loader import *
# from unet_model import *
# from tqdm.auto import tqdm
# from timeit import default_timer as timer

# train_dataloader, test_dataloader = get_dataloaders(
#     blurred_dir=train_dir_blurred,
#     sharp_dir=train_dir_sharp,
#     train_transforms=train_transforms,
#     test_transforms=test_transforms,
#     batch_size=64
# )

# # 1. Getting one batch from train dataloader
# blurred_batch, sharp_batch = next(iter(train_dataloader))


# # 2. Get a single blurred image + its sharp target
# blurred_single = blurred_batch[0].unsqueeze(dim=0)  # shape: [1, 3, H, W]
# sharp_single   = sharp_batch[0]                      # shape: [3, H, W]

# model_0.eval()
# device= torch.device("cuda" if torch.cuda.is_available() else "cpu")
# model_0=model_0.to(device)

# blurred_single = blurred_single.to(device)
# sharp_single   = sharp_single.to(device)
# with torch.inference_mode():
#     output =model_0(blurred_single)

# # print(f"Input (blurred) shape: {blurred_single.shape}")
# # print(f"Target (sharp) shape: {sharp_single.shape}")
# # print(f"Model output shape:   {output.shape}")

# # summary(model_0, input_size=[1, 3, 128, 128]) # do a test pass through of an example input size

# # train step 


# def psnr(y_true: torch.Tensor, y_pred: torch.Tensor, max_val: float = 1.0):
#     """Compute PSNR for a batch of images.
#     y_true and y_pred should be in range [0, 1].
#     """

#     y_true = (y_true + 1) / 2
#     y_pred = (y_pred + 1) / 2
#     mse = F.mse_loss(y_pred, y_true, reduction='mean')
#     if mse == 0:
#         return torch.tensor(100.0)  # perfect match
#     psnr_val = 20 * torch.log10(max_val / torch.sqrt(mse))
#     return psnr_val

# def train_step(model: torch.nn.Module, 
#                dataloader: torch.utils.data.DataLoader, 
#                loss_fn: torch.nn.Module, 
#                optimizer: torch.optim.Optimizer,
#                device: torch.device):
#     """Single epoch training step for image-to-image regression with PSNR metric."""
    
#     model.train()
#     train_loss = 0.0
#     train_psnr = 0.0

#     for X, y in tqdm(dataloader, desc="Train batches", leave=False):
#         X, y = X.to(device), y.to(device)

#         # Forward pass
#         y_pred = model(X)

#         # Compute loss
#         loss = loss_fn(y_pred, y)
#         train_loss += loss.item()

#         # Backpropagation
#         optimizer.zero_grad()
#         loss.backward()
#         optimizer.step()

#         # Compute PSNR for this batch
#         batch_psnr = psnr(y, y_pred)
#         train_psnr += batch_psnr.item()

#     # Average over batches
#     train_loss /= len(dataloader)
#     train_psnr /= len(dataloader)
    
#     return train_loss, train_psnr


# #test_step

# def test_step(model: torch.nn.Module, 
#               dataloader: torch.utils.data.DataLoader, 
#               loss_fn: torch.nn.Module,
#               device: torch.device):
#     """Single epoch testing/validation step with PSNR metric."""
    
#     model.eval()
#     test_loss = 0.0
#     test_psnr = 0.0
    
#     with torch.inference_mode():
#         for batch, (X, y) in enumerate(dataloader):
#             X, y = X.to(device), y.to(device)
            
#             # Forward pass
#             y_pred = model(X)
            
#             # Compute loss
#             loss = loss_fn(y_pred, y)
#             test_loss += loss.item()
            
#             # Compute PSNR
#             batch_psnr = psnr(y, y_pred)
#             test_psnr += batch_psnr.item()
    
#     # Average over all batches
#     test_loss /= len(dataloader)
#     test_psnr /= len(dataloader)
    
#     return test_loss, test_psnr

# #train loop

# def train(model, train_dataloader, test_dataloader, optimizer, loss_fn=None, epochs=1, device='cuda'):
#     if loss_fn is None:
#         loss_fn = nn.MSELoss()

#     model.to(device)

#     results = {
#         "train_loss": [],
#         "train_psnr":[],
#         "test_loss": [],
#         "test_psnr": []
#     }

#     sample_X, sample_y = next(iter(train_dataloader))
#     sample_X, sample_y = sample_X.to(device), sample_y.to(device)

#     torch.cuda.synchronize()  # clear any pending ops
#     start = time.time()
#     _ = model(sample_X)       # forward pass
#     torch.cuda.synchronize()
#     batch_time = time.time() - start

#     batches_per_epoch = len(train_dataloader)
#     est_epoch_time = batch_time * batches_per_epoch
#     est_total_time = est_epoch_time * epochs

#     print(f"\nEstimated time per epoch: {est_epoch_time/60:.2f} minutes")
#     print(f"Estimated total training time ({epochs} epochs): {est_total_time/60:.2f} minutes")
#     print("="*60)


#     for epoch in tqdm(range(epochs), desc="Training"):
#         train_loss, train_psnr = train_step(model, train_dataloader, loss_fn, optimizer, device)
#         test_loss, test_psnr = test_step(model, test_dataloader, loss_fn, device)

#         print(f"Epoch [{epoch+1}/{epochs}] "
#               f"Train Loss: {train_loss:.4f} | "
#               f"Train PSNR: {train_psnr:.4f} | "
#               f"Test Loss: {test_loss:.4f} | "
#               f"Test PSNR: {test_psnr:.2f} dB")

#         results["train_loss"].append(train_loss)
#         results["train_psnr"].append(train_psnr)
#         results["test_loss"].append(test_loss)
#         results["test_psnr"].append(test_psnr)

#     return results

# if __name__ == "__main__":
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#     # -------------------------
#     # Set random seeds
#     # -------------------------
#     torch.manual_seed(42)
#     torch.cuda.manual_seed(42)

#     # -------------------------
#     # Hyperparameters
#     # -------------------------
#     NUM_EPOCHS = 200
#     LEARNING_RATE = 1e-4

#     # -------------------------
#     # Model
#     # -------------------------
#     model_0 = UNet(n_channels=3, n_classes=3).to(device)

#     # -------------------------
#     # Loss function and optimizer
#     # -------------------------
#     loss_fn = nn.MSELoss()
#     optimizer = torch.optim.Adam(model_0.parameters(), lr=LEARNING_RATE)

#     # -------------------------
#     # Start training
#     # -------------------------
#     start_time = timer()

#     results = train(
#         model=model_0,
#         train_dataloader=train_dataloader,
#         test_dataloader=test_dataloader,
#         optimizer=optimizer,
#         loss_fn=loss_fn,
#         epochs=NUM_EPOCHS
#     )

#     end_time = timer()
#     print(f"\nTotal training time: {end_time-start_time:.2f} seconds")

# torch.save(model_0.state_dict(), "unet_model.pth")
# torch.save(results, "results.pth")
# train.py
import torch
from torch import nn, optim
from data_loader import get_dataloaders
from data_augmentation import PairedTransforms
from preprocessing import *
from unet_model import UNet
from train_utils import train_model , charbonnier_loss
# from metrics_utils import lpips_fn
import torch
from train_utils import estimate_training_time
import time



DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

train_transforms = PairedTransforms(resize=(128,128))
test_transforms = PairedTransforms(resize=(128,128), flip_prob=0.0)

train_loader, test_loader = get_dataloaders(
    blurred_dir=train_dir_blurred,
    sharp_dir= train_dir_sharp,
    train_transforms=train_transforms,
    test_transforms=test_transforms,
    batch_size=64
)

model = UNet(n_channels=3, n_classes=3).to(DEVICE)
optimizer = optim.Adam(model.parameters(), lr=1e-4)
# loss_fn = lambda y_pred, y_true: combined_loss(
#     y_pred, y_true,
#     alpha=1,  # weight for L1
#     beta=1,   # weight for SSIM
#     gamma=0.1,  # weight for LPIPS
#     lpips_fn=lpips_fn
# )
loss_fn = charbonnier_loss

start_time = time.time()
estimate_training_time(model, train_loader, test_loader, loss_fn, optimizer, DEVICE, epochs=200)
results = train_model(model, train_loader, test_loader, optimizer, loss_fn, epochs=200, device=DEVICE)

end_time = time.time()
total_time = end_time - start_time

hours, rem = divmod(total_time, 3600)
minutes, seconds = divmod(rem, 60)
print(f"\nTotal training time: {int(hours)}h {int(minutes)}m {seconds:.2f}s")

torch.save(model.state_dict(), "unet_model_v2.pth")
torch.save(results, "results_v2.pth")
