import random
import matplotlib.pyplot as plt
import torch

def display_random_image_pairs(dataset, n=5, seed=None):
    """
    Displays n random pairs of blurred and sharp images from a paired dataset.

    Args:
        dataset (torch.utils.data.Dataset): Dataset returning (blurred, sharp) image pairs.
        n (int): Number of image pairs to display.
        seed (int, optional): Random seed for reproducibility.
    """
    if seed:
        random.seed(seed)
    
    # Limit n for display purposes
    n = min(n, 10)
    
    # Sample random indices
    samples = random.sample(range(len(dataset)), k=n) #pulls out random n samples
    
    # Setup figure
    plt.figure(figsize=(16, 8))
    
    for i, idx in enumerate(samples): #here i is the index 0,1,... and idx will be the values of index it takes like if samples =[43,25,34] then i will be 0,1,2 and idx will be 43, 25, 34
        blurred, sharp = dataset[idx] #
        
        # Convert from [C, H, W] -> [H, W, C] for matplotlib
        blurred = blurred.permute(1, 2, 0)
        sharp   = sharp.permute(1, 2, 0)
        
        # Plot blurred image
        plt.subplot(2, n, i+1)
        plt.imshow(blurred)
        plt.title("Blurred")
        plt.axis("off")
        
        # Plot corresponding sharp image
        plt.subplot(2, n, i+1+n)
        plt.imshow(sharp)
        plt.title("Sharp")
        plt.axis("off")
    
    plt.tight_layout()
    plt.show()
