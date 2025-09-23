import os
import pathlib
import torch

from data_augmentation import *
from helper_functions import *
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
from torch.utils.data import DataLoader
from typing import Tuple,Dict, List
from torch.utils.data import random_split



train_transforms = PairedTransforms(resize=(64, 64))
test_transforms  = PairedTransforms(resize=(64, 64))  # usually no flip for test
class DeblurDataset(Dataset):
    def __init__(self, blurred_dir, sharp_dir, transform=None):
        self.blurred_dir = blurred_dir
        self.sharp_dir = sharp_dir
        self.transform = transform

        # Match images by sorting
        self.blurred_images = sorted(os.listdir(blurred_dir))
        self.sharp_images = sorted(os.listdir(sharp_dir))

        assert len(self.blurred_images) == len(self.sharp_images), \
            "Blurred and sharp folders must contain the same number of images"

    def __len__(self):
        return len(self.blurred_images)

    def __getitem__(self, idx):
        blurred_path = os.path.join(self.blurred_dir, self.blurred_images[idx])
        sharp_path   = os.path.join(self.sharp_dir,   self.sharp_images[idx])

        blurred_img = Image.open(blurred_path).convert("RGB")
        sharp_img   = Image.open(sharp_path).convert("RGB")

        if self.transform:
            # blurred_img = self.transform(blurred_img)
            # sharp_img   = self.transform(sharp_img)
            blurred_img, sharp_img = self.transform(blurred_img, sharp_img)

        return blurred_img, sharp_img

# train_dataset = DeblurDataset(
#     blurred_dir=train_dir_blurred,
#     sharp_dir=train_dir_sharp,
#     transform=train_transforms
# )

# # If you have a separate test folder:
# test_dataset = DeblurDataset(
#     blurred_dir=test_dir_blurred,
#     sharp_dir=test_dir_sharp,
#     transform=test_transforms
# )


#i dont have a separate test datset so i am splitting training dataset into 80% and 20% fro testing



# Full dataset from your 1 lakh images
# full_dataset = DeblurDataset(
#     blurred_dir=train_dir_blurred,
#     sharp_dir=train_dir_sharp,
#     transform=train_transforms
# )

# # Define split sizes
# train_size = int(0.8 * len(full_dataset))   # 80% train → 80,000
# test_size  = len(full_dataset) - train_size # 20% test → 20,000

# # Perform the split
# train_dataset, test_dataset = random_split(full_dataset, [train_size, test_size])


# #data loading
# BATCH_SIZE = 64
# NUM_WORKERS = os.cpu_count()
# train_dataloader= DataLoader(train_dataset,
#                             batch_size=BATCH_SIZE,
#                             num_workers=NUM_WORKERS,
#                             shuffle=True)

# test_dataloader = DataLoader(test_dataset,
#                              batch_size=BATCH_SIZE,
#                              num_workers=NUM_WORKERS,
#                              shuffle=False)


class SubsetWithTransform(Dataset):
    """
    Wraps a subset of a dataset and applies a specific transform.
    Works with PairedTransforms that expect (img_blur, img_sharp)
    """
    def __init__(self, subset, transform):
        self.subset = subset
        self.transform = transform

    def __len__(self):
        return len(self.subset)

    def __getitem__(self, idx):
        img_blur, img_sharp = self.subset[idx]  # get both images from subset
        if self.transform:
            img_blur, img_sharp = self.transform(img_blur, img_sharp)
        return img_blur, img_sharp


def get_dataloaders(blurred_dir, sharp_dir, train_transforms, test_transforms,
                    batch_size=64, split_ratio=0.8):
    """
    Returns train and test dataloaders from a single dataset folder.
    """
    # Full dataset with default transforms (can be train_transforms)
    full_dataset = DeblurDataset(blurred_dir, sharp_dir, transform=None)

    # Split sizes
    train_size = int(split_ratio * len(full_dataset))
    test_size  = len(full_dataset) - train_size

    # Random split
    train_subset, test_subset = random_split(full_dataset, [train_size, test_size])

    # Wrap subsets with specific transforms
    train_dataset = SubsetWithTransform(train_subset, train_transforms)
    test_dataset  = SubsetWithTransform(test_subset, test_transforms)

    # Number of workers
    num_workers = os.cpu_count()

    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    test_loader  = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return train_loader, test_loader

