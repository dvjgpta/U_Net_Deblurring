import torch
import random
import matplotlib.pyplot as plt
from preprocessing import *
from PIL import Image
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# # Write transform for image
# data_transform = transforms.Compose([
#     # Resize the images to 64x64
#     transforms.Resize(size=(64, 64)),
#     # Flip the images randomly on the horizontal
#     transforms.RandomHorizontalFlip(p=0.5), # p = probability of flip, 0.5 = 50% chance
#     # Turn the image into a torch.Tensor
#     transforms.ToTensor() # this also converts all pixel values from 0 to 255 to be between 0.0 and 1.0 
# ])


# def plot_transformed_images(image_paths, transform, n=3, seed=42):
#     """Plots a series of random images from image_paths.

#     Will open n image paths from image_paths, transform them
#     with transform and plot them side by side.

#     Args:
#         image_paths (list): List of target image paths. 
#         transform (PyTorch Transforms): Transforms to apply to images.
#         n (int, optional): Number of images to plot. Defaults to 3.
#         seed (int, optional): Random seed for the random generator. Defaults to 42.
#     """
#     random.seed(seed)
#     random_image_paths = random.sample(image_paths, k=n)
#     for image_path in random_image_paths:
#         with Image.open(image_path) as f:
#             fig, ax = plt.subplots(1, 2)
#             ax[0].imshow(f) 
#             ax[0].set_title(f"Original \nSize: {f.size}")
#             ax[0].axis("off")

#             # Transform and plot image
#             # Note: permute() will change shape of image to suit matplotlib 
#             # (PyTorch default is [C, H, W] but Matplotlib is [H, W, C])
#             transformed_image = transform(f).permute(1, 2, 0) 
#             ax[1].imshow(transformed_image) 
#             ax[1].set_title(f"Transformed \nSize: {transformed_image.shape}")
#             ax[1].axis("off")

#             fig.suptitle(f"Class: {image_path.parent.stem}", fontsize=16)
#             plt.show()

# plot_transformed_images(image_path_list, 
#                         transform=data_transform, 
#                         n=3)


# Augment train data
class PairedTransforms:
    def __init__(self, resize=(64, 64), flip_prob=0.5):
        self.resize = transforms.Resize(resize)
        self.to_tensor = transforms.ToTensor()
        self.flip_prob = flip_prob

    def __call__(self, img_blur, img_sharp):
        # Resize both
        img_blur = self.resize(img_blur)
        img_sharp = self.resize(img_sharp)

        # Random horizontal flip
        if random.random() < self.flip_prob:
            img_blur = transforms.functional.hflip(img_blur)
            img_sharp = transforms.functional.hflip(img_sharp)

        # Convert to tensor
        return self.to_tensor(img_blur), self.to_tensor(img_sharp)
