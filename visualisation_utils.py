# visualization_utils.py
import torch
import matplotlib.pyplot as plt
from metrics_utils import denormalize

def visualize_predictions(model, dataloader, device='cuda', num_images=5, normalize_range='0_to_1'):
    model.eval()
    device = torch.device(device if torch.cuda.is_available() else 'cpu')
    X, y = next(iter(dataloader))
    X, y = X.to(device), y.to(device)
    with torch.inference_mode():
        preds = model(X)

    X = X.cpu().permute(0,2,3,1)
    y = y.cpu().permute(0,2,3,1)
    preds = preds.cpu().permute(0,2,3,1)

    X = denormalize(X, normalize_range)
    y = denormalize(y, normalize_range)
    preds = denormalize(preds, normalize_range)

    fig, axes = plt.subplots(3, num_images, figsize=(15,5))
    for i in range(num_images):
        axes[0,i].imshow(X[i]); axes[0,i].axis('off'); axes[0,i].set_title("Blurred")
        axes[1,i].imshow(y[i]); axes[1,i].axis('off'); axes[1,i].set_title("Sharp GT")
        axes[2,i].imshow(preds[i]); axes[2,i].axis('off'); axes[2,i].set_title("Predicted")
    plt.tight_layout()
    plt.show()


import torch
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt

# def custom_test(model, image_path, device, resize=(64,64)):
#     """
#     Run a trained model on a single external blurred image
#     and visualize input vs deblurred output.

#     Args:
#         model: Trained UNet model
#         image_path: Path to blurred input image
#         device: torch.device("cuda" or "cpu")
#         resize: tuple, resize for consistency (default 64x64)

#     Returns:
#         output_img (PIL.Image): Deblurred output image
#     """

#     # Put model in eval mode
#     model.eval()

#     # Define preprocessing (same as training/test transforms)
#     transform = transforms.Compose([
#         transforms.Resize(resize),
#         transforms.ToTensor()
#     ])

#     # Load and preprocess image
#     img = Image.open(image_path).convert("RGB")
#     input_tensor = transform(img).unsqueeze(0).to(device)  # shape (1,3,H,W)

#     with torch.no_grad():
#         output = model(input_tensor)
    
#     # Postprocess: tensor → image
#     output = output.squeeze(0).detach().cpu()
#     output = (output.clamp(0,1) * 255).byte()  # scale back to [0,255]
#     output_img = transforms.ToPILImage()(output)

#     # Show side-by-side
#     fig, axs = plt.subplots(1,2, figsize=(6,3))
#     axs[0].imshow(img)
#     axs[0].set_title("Blurred Input")
#     axs[0].axis("off")

#     axs[1].imshow(output_img)
#     axs[1].set_title("Deblurred Output")
#     axs[1].axis("off")

#     plt.show()

#     return output_img

def custom_test(model, image_path, device, patch_size=64, overlap=16, save_path=None):
    """
    Run patch-wise inference on a high-res external image.
    
    Args:
        model: Trained UNet
        image_path: Path to input image
        device: 'cuda' or 'cpu'
        patch_size: Size of patches (64, since your model was trained on that)
        overlap: Overlap between patches to reduce seams
        save_path: Where to save the deblurred image (optional)
    """
    model.eval()
    model.to(device)

    # Load and convert to tensor [C,H,W], range [0,1]
    img = Image.open(image_path).convert("RGB")
    orig_w, orig_h = img.size
    transform = transforms.ToTensor()
    img_tensor = transform(img).unsqueeze(0).to(device)  # [1,3,H,W]

    # Pad image to be divisible by patch_size - overlap
    stride = patch_size - overlap
    pad_h = (stride - (orig_h % stride)) % stride
    pad_w = (stride - (orig_w % stride)) % stride
    img_tensor = torch.nn.functional.pad(img_tensor, (0,pad_w,0,pad_h), mode='replicate')

    _, _, H, W = img_tensor.shape
    output = torch.zeros_like(img_tensor)

    # Weight mask for blending overlaps
    weight = torch.zeros_like(img_tensor)

    # Slide window
    for y in range(0, H - patch_size + 1, stride):
        for x in range(0, W - patch_size + 1, stride):
            patch = img_tensor[:, :, y:y+patch_size, x:x+patch_size]
            with torch.inference_mode():
                pred_patch = model(patch)

            output[:, :, y:y+patch_size, x:x+patch_size] += pred_patch
            weight[:, :, y:y+patch_size, x:x+patch_size] += 1

    # Normalize overlapping regions
    output /= weight

    # Crop back to original size
    output = output[:, :, :orig_h, :orig_w]

    # Convert back to PIL Image
    out_img = output.squeeze(0).detach().cpu().clamp(0,1)
    out_img = transforms.ToPILImage()(out_img)

    if save_path:
        out_img.save(save_path)

    return out_img