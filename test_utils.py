# test_utils.py
import torch
from train_utils import psnr  # your PSNR function
from metrics_utils import compute_ssim, compute_lpips

# Create LPIPS model once


def test_model(model, dataloader, loss_fn, device):
    """
    Evaluate a model on a dataset.

    Returns:
        avg_loss, avg_psnr, avg_ssim, avg_lpips
    """
    model.eval()
    total_loss = 0.0
    total_psnr = 0.0
    total_ssim = 0.0
    total_lpips = 0.0

    device = torch.device(device if torch.cuda.is_available() else 'cpu')

    with torch.inference_mode():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            y_pred = model(X)

            # Loss
            total_loss += loss_fn(y_pred, y).item()
            # PSNR
            total_psnr += psnr(y, y_pred).item()
            # SSIM
            total_ssim += compute_ssim(y, y_pred)
            # LPIPS
            total_lpips += compute_lpips(y, y_pred)

    n = len(dataloader)
    avg_loss = total_loss / n
    avg_psnr = total_psnr / n
    avg_ssim = total_ssim / n
    avg_lpips = total_lpips / n

    print(f"Test Loss: {avg_loss:.4f} | PSNR: {avg_psnr:.2f} dB | SSIM: {avg_ssim:.4f} | LPIPS: {avg_lpips:.4f}")
    return avg_loss, avg_psnr, avg_ssim, avg_lpips
