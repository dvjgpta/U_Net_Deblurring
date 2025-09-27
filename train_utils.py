# # train_utils.py
# import torch
# import torch.nn.functional as F
# from tqdm.auto import tqdm

# def psnr(y_true, y_pred, max_val=1.0):
#     y_true = (y_true + 1)/2
#     y_pred = (y_pred + 1)/2
#     mse = F.mse_loss(y_pred, y_true, reduction='mean')
#     if mse == 0:
#         return torch.tensor(100.0)
#     return 20 * torch.log10(max_val / torch.sqrt(mse))

# def train_step(model, dataloader, loss_fn, optimizer, device):
#     model.train()
#     total_loss, total_psnr = 0, 0
#     for X, y in tqdm(dataloader, leave=False):
#         X, y = X.to(device), y.to(device)
#         optimizer.zero_grad()
#         y_pred = model(X)
#         loss = loss_fn(y_pred, y)
#         loss.backward()
#         optimizer.step()
#         total_loss += loss.item()
#         total_psnr += psnr(y, y_pred).item()
#     return total_loss/len(dataloader), total_psnr/len(dataloader)

# from metrics_utils import compute_ssim, compute_lpips

# def test_step(model, dataloader, loss_fn, device='cuda'):
#     model.eval()
#     test_loss = 0.0
#     test_psnr = 0.0
#     test_ssim = 0.0
#     test_lpips = 0.0
    
#     with torch.inference_mode():
#         for X, y in dataloader:
#             X, y = X.to(device), y.to(device)
            
#             y_pred = model(X)
            
#             # Loss
#             loss = loss_fn(y_pred, y)
#             test_loss += loss.item()
            
#             # Metrics
#             test_psnr += psnr(y, y_pred).item()
#             test_ssim += compute_ssim(y, y_pred).item()
#             test_lpips += compute_lpips(y, y_pred).item()
    
#     n = len(dataloader)
#     return test_loss/n, test_psnr/n, test_ssim/n, test_lpips/n

# def train_model(model, train_loader, test_loader, optimizer, loss_fn, epochs, device):
#     results = {"train_loss": [], "train_psnr": [], "test_loss": [], "test_psnr": []}
#     model.to(device)
#     for epoch in range(epochs):
#         train_loss, train_psnr = train_step(model, train_loader, loss_fn, optimizer, device)
#         test_loss, test_psnr = test_step(model, test_loader, loss_fn, device)
#         print(f"Epoch [{epoch+1}/{epochs}] Train Loss: {train_loss:.4f} | Train PSNR: {train_psnr:.2f} | Test Loss: {test_loss:.4f} | Test PSNR: {test_psnr:.2f}")
#         results["train_loss"].append(train_loss)
#         results["train_psnr"].append(train_psnr)
#         results["test_loss"].append(test_loss)
#         results["test_psnr"].append(test_psnr)
#     return results

# train_utils.py
# import torch
# import torch.nn.functional as F
# from tqdm.auto import tqdm
# from metrics_utils import compute_ssim, compute_lpips

# def psnr(y_true, y_pred, max_val=1.0):
#     """
#     Compute PSNR for a batch of images.
#     Expects input in range [-1,1] or [0,1].
#     """
#     # Convert [-1,1] to [0,1]
#     if y_true.min() < 0:
#         y_true = (y_true + 1)/2
#     if y_pred.min() < 0:
#         y_pred = (y_pred + 1)/2
#     mse = F.mse_loss(y_pred, y_true, reduction='mean')
#     if mse == 0:
#         return torch.tensor(100.0)
#     return 20 * torch.log10(max_val / torch.sqrt(mse))

# def train_step(model, dataloader, loss_fn, optimizer, device):
#     model.train()
#     total_loss, total_psnr, total_ssim, total_lpips = 0, 0, 0, 0
#     for X, y in tqdm(dataloader, leave=False, desc="Train Batches"):
#         X, y = X.to(device), y.to(device)
#         optimizer.zero_grad()
#         y_pred = model(X)
#         loss = loss_fn(y_pred, y)
#         loss.backward()
#         optimizer.step()

#         total_loss += loss.item()
#         total_psnr += psnr(y, y_pred).item()
#         total_ssim += compute_ssim(y, y_pred)
#         total_lpips += compute_lpips(y, y_pred)

#     n = len(dataloader)
#     return total_loss/n, total_psnr/n, total_ssim/n, total_lpips/n

# def test_step(model, dataloader, loss_fn, device):
#     model.eval()
#     total_loss, total_psnr, total_ssim, total_lpips = 0, 0, 0, 0
#     device = torch.device(device if torch.cuda.is_available() else 'cpu')

#     with torch.inference_mode():
#         for X, y in tqdm(dataloader, leave=False, desc="Test Batches"):
#             X, y = X.to(device), y.to(device)
#             y_pred = model(X)

#             total_loss += loss_fn(y_pred, y).item()
#             total_psnr += psnr(y, y_pred).item()
#             total_ssim += compute_ssim(y, y_pred)
#             total_lpips += compute_lpips(y, y_pred)

#     n = len(dataloader)
#     return total_loss/n, total_psnr/n, total_ssim/n, total_lpips/n

# def train_model(model, train_loader, test_loader, optimizer, loss_fn, epochs, device):
#     results = {
#         "train_loss": [], "train_psnr": [], "train_ssim": [], "train_lpips": [],
#         "test_loss": [], "test_psnr": [], "test_ssim": [], "test_lpips": []
#     }

#     model.to(device)

#     for epoch in range(epochs):
#         print(f"\nEpoch [{epoch+1}/{epochs}]")
#         train_loss, train_psnr, train_ssim, train_lpips = train_step(model, train_loader, loss_fn, optimizer, device)
#         test_loss, test_psnr, test_ssim, test_lpips = test_step(model, test_loader, loss_fn, device)

#         print(f"Train -> Loss: {train_loss:.4f}, PSNR: {train_psnr:.2f}, SSIM: {train_ssim:.4f}, LPIPS: {train_lpips:.4f}")
#         print(f"Test  -> Loss: {test_loss:.4f}, PSNR: {test_psnr:.2f}, SSIM: {test_ssim:.4f}, LPIPS: {test_lpips:.4f}")

#         results["train_loss"].append(train_loss)
#         results["train_psnr"].append(train_psnr)
#         results["train_ssim"].append(train_ssim)
#         results["train_lpips"].append(train_lpips)
#         results["test_loss"].append(test_loss)
#         results["test_psnr"].append(test_psnr)
#         results["test_ssim"].append(test_ssim)
#         results["test_lpips"].append(test_lpips)

#     return results

import torch
import torch.nn.functional as F
from tqdm.auto import tqdm
# from metrics_utils import compute_ssim, compute_lpips

# ------------------------------
# PSNR calculation
# ------------------------------
def psnr(y_true, y_pred, max_val=1.0):
    """
    Compute PSNR for a batch of images.
    Expects input in range [-1,1] or [0,1].
    """
    # Convert [-1,1] to [0,1]
    if y_true.min() < 0:
        y_true = (y_true + 1)/2
    if y_pred.min() < 0:
        y_pred = (y_pred + 1)/2

    mse = F.mse_loss(y_pred, y_true, reduction='mean')
    if mse == 0:
        return torch.tensor(100.0)
    return 20 * torch.log10(max_val / torch.sqrt(mse))

# def combined_loss(y_pred, y_true, alpha=0.8, beta=0.1, gamma=0.1, lpips_fn=None):
#     l1 = F.l1_loss(y_pred, y_true)
    
#     ssim_val = compute_ssim(y_pred, y_true)
#     ssim_loss = 1 - torch.tensor(ssim_val, device=y_pred.device)
    
#     if lpips_fn is not None:
#         lpips_fn = lpips_fn.to(y_pred.device)
#         lpips_loss = lpips_fn(y_pred, y_true).mean()
#     else:
#         lpips_loss = 0.0

#     return alpha * l1 + beta * ssim_loss + gamma * lpips_loss



def charbonnier_loss(y_pred, y_true, epsilon=1e-6):
    """
    Charbonnier loss: smooth L1 variant
    """
    diff = y_pred - y_true
    return torch.mean(torch.sqrt(diff * diff + epsilon**2))

# def combined_loss(y_pred, y_true, alpha=0.8, beta=0.1, gamma=0.1, lpips_fn=None):
#     l1 = charbonnier_loss(y_pred, y_true)

#     # SSIM part
#     ssim_val = compute_ssim(y_pred, y_true)
#     ssim_loss = 1 - torch.tensor(ssim_val, device=y_pred.device)

#     # LPIPS part
#     if lpips_fn is not None:
#         lpips_fn = lpips_fn.to(y_pred.device)
#         lpips_loss = lpips_fn(y_pred, y_true).mean()
#     else:
#         lpips_loss = 0.0

#     return alpha * l1 + beta * ssim_loss + gamma * lpips_loss

def psnr_focused_loss(y_pred, y_true, alpha=1.0, beta=1.0, epsilon=1e-6):
    """
    Combined loss: Charbonnier + L2 for PSNR improvement.
    
    Args:
        y_pred (torch.Tensor): Predicted tensor
        y_true (torch.Tensor): Ground truth tensor
        alpha (float): Weight for Charbonnier loss
        beta (float): Weight for L2 loss
        epsilon (float): Small value for Charbonnier stability
    """
    loss_char = charbonnier_loss(y_pred, y_true, epsilon)
    loss_l2   = F.mse_loss(y_pred, y_true)
    return alpha * loss_char + beta * loss_l2


import time

def estimate_training_time(model, train_loader, test_loader, loss_fn, optimizer, device, epochs=1, num_batches=10):
    """
    Estimate training time including testing.
    Uses a representative sample of batches (num_batches) for scaling.
    """
    model.to(device)
    
    # Warmup pass (ignore timing)
    X, y = next(iter(train_loader))
    X, y = X.to(device), y.to(device)
    _ = model(X)

    # ---- Training timing ----
    model.train()
    start = time.time()
    for i, (X, y) in enumerate(train_loader):
        if i >= num_batches:
            break
        X, y = X.to(device), y.to(device)
        optimizer.zero_grad()
        y_pred = model(X)
        loss = loss_fn(y_pred, y)
        loss.backward()
        optimizer.step()
    train_time_sample = time.time() - start
    train_time_per_batch = train_time_sample / num_batches

    # ---- Testing timing ----
    model.eval()
    start = time.time()
    with torch.inference_mode():
        for i, (X, y) in enumerate(test_loader):
            if i >= num_batches:
                break
            X, y = X.to(device), y.to(device)
            y_pred = model(X)
            _ = loss_fn(y_pred, y)
            _ = psnr(y, y_pred)  # only PSNR, skip heavy metrics
    test_time_sample = time.time() - start
    test_time_per_batch = test_time_sample / num_batches

    # ---- Scaling to epoch ----
    # account for data loading & optimizations -> use correction factor
    correction_factor = 0.65   # ~35-40% faster after warmup
    total_batches_train = len(train_loader)
    total_batches_test = len(test_loader)

    time_per_epoch = (
        (train_time_per_batch * total_batches_train) +
        (test_time_per_batch * total_batches_test)
    ) * correction_factor

    total_time_est = time_per_epoch * epochs

    print(f"Estimated time per epoch: {time_per_epoch/60:.2f} min")
    print(f"Estimated total training time for {epochs} epochs: {total_time_est/3600:.2f} hours")

    return total_time_est


# ------------------------------
# Train step (loss only)
# ------------------------------
def train_step(model, dataloader, loss_fn, optimizer, device):
    model.train()
    total_loss = 0

    for X, y in tqdm(dataloader, leave=False, desc="Train Batches"):
        X, y = X.to(device), y.to(device)

        optimizer.zero_grad()
        y_pred = model(X)

        loss = loss_fn(y_pred, y)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(dataloader)

# ------------------------------
# Test/validation step (compute metrics)
# ------------------------------
def test_step(model, dataloader, loss_fn, device):
    model.eval()
    total_loss = 0
    all_psnr, all_ssim, all_lpips = [], [], []

    with torch.inference_mode():
        for X, y in tqdm(dataloader, leave=False, desc="Test Batches"):
            X, y = X.to(device), y.to(device)
            y_pred = model(X)

            total_loss += loss_fn(y_pred, y).item()

            # Metrics (can be heavy, but only computed per epoch on test set)
            all_psnr.append(psnr(y, y_pred).item())
            # all_ssim.append(compute_ssim(y, y_pred))
            # all_lpips.append(compute_lpips(y, y_pred))

    n = len(dataloader)
    return {
        "loss": total_loss / n,
        "psnr": sum(all_psnr) / n,
        # "ssim": sum(all_ssim) / n,
        # "lpips": sum(all_lpips) / n
    }

# ------------------------------
# Full training loop
# ------------------------------
def train_model(model, train_loader, test_loader, optimizer, loss_fn, epochs, device):
    results = {
        "train_loss": [],
        "test_loss": [],
        "test_psnr": [],
        # "test_ssim": [],
        # "test_lpips": []
    }

    model.to(device)

    for epoch in range(epochs):
        print(f"\nEpoch [{epoch+1}/{epochs}]")

        # Training
        train_loss = train_step(model, train_loader, loss_fn, optimizer, device)

        # Validation / Metrics
        test_metrics = test_step(model, test_loader, loss_fn, device)

        # Print
        print(f"Train Loss: {train_loss:.4f}")
        print(f"Test -> Loss: {test_metrics['loss']:.4f}, PSNR: {test_metrics['psnr']:.2f}, "
            #   f"SSIM: {test_metrics['ssim']:.4f}, LPIPS: {test_metrics['lpips']:.4f}"
            )

        # Save results
        results["train_loss"].append(train_loss)
        results["test_loss"].append(test_metrics["loss"])
        results["test_psnr"].append(test_metrics["psnr"])
        # results["test_ssim"].append(test_metrics["ssim"])
        # results["test_lpips"].append(test_metrics["lpips"])

    return results
