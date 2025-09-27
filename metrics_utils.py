# metrics_utils.py
import torch
import matplotlib.pyplot as plt
from pytorch_msssim import ssim
import lpips

# Create LPIPS model
lpips_fn = lpips.LPIPS(net='alex').to('cuda')  # or 'cpu'

# lpips_fn = lpips.LPIPS(net='alex').to('cpu')  # move LPIPS to CPU

def compute_lpips(y_true, y_pred):
    """
    y_true, y_pred: tensors in range [-1,1], shape [B,3,H,W]
    Returns average LPIPS over batch
    """
    # Ensure [-1,1] for LPIPS
    y_true = y_true if y_true.min() < 0 else y_true*2 - 1
    y_pred = y_pred if y_pred.min() < 0 else y_pred*2 - 1
    
    return lpips_fn(y_pred, y_true).mean()

# def compute_lpips(y_true, y_pred):
#     """
#     y_true, y_pred: tensors in range [-1,1], shape [B,3,H,W]
#     Returns average LPIPS over batch
#     """
#     # Ensure [-1,1] for LPIPS
#     y_true = y_true if y_true.min() < 0 else y_true*2 - 1
#     y_pred = y_pred if y_pred.min() < 0 else y_pred*2 - 1
    
#     # Move to CPU
#     y_true_cpu = y_true.detach().cpu()
#     y_pred_cpu = y_pred.detach().cpu()
    
#     return lpips_fn(y_pred_cpu, y_true_cpu).mean()


def compute_ssim(y_true, y_pred):
    """
    y_true, y_pred: tensors in range [0,1] or [-1,1]
    Returns average SSIM over batch
    """
    # Ensure [0,1] range
    y_true = (y_true + 1)/2 if y_true.min() < 0 else y_true
    y_pred = (y_pred + 1)/2 if y_pred.min() < 0 else y_pred
    
    return ssim(y_pred, y_true, data_range=1.0, size_average=True)

def count_parameters(model):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total params: {total}, Trainable params: {trainable}")

def plot_training_metrics(results):
    epochs = list(range(1, len(results['train_loss'])+1))
    for key in results:
        results[key] = [float(x.cpu()) if torch.is_tensor(x) else float(x) for x in results[key]]
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Plot losses
    axes[0].plot(epochs, results['train_loss'], label='Train Loss')
    axes[0].plot(epochs, results['test_loss'], label='Test Loss')
    axes[0].set_title("Loss over Epochs")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].legend()

    # Plot metrics only for test
    # axes[1].plot(epochs, results['test_ssim'], label='Test SSIM')
    # axes[1].plot(epochs, results['test_lpips'], label='Test LPIPS')
    # axes[1].set_title("Test Metrics over Epochs")
    # axes[1].set_xlabel("Epoch")
    # axes[1].set_ylabel("Value")
    # axes[1].legend()

    # Plot PSNR
    axes[1].plot(epochs,results['test_psnr'], label='Test PSNR', color='green')
    axes[1].set_title("Test PSNR over Epochs")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("PSNR (dB)")
    axes[1].legend()

    plt.tight_layout()
    plt.show()

def denormalize(tensor, range='-1_to_1'):
    if range=='-1_to_1':
        return torch.clamp((tensor+1)/2,0,1)
    elif range=='0_to_1':
        return torch.clamp(tensor,0,1)
    else:
        raise ValueError("range must be '-1_to_1' or '0_to_1'")
