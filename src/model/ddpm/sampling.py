import torch
import pickle 
import os
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset
from torchvision import datasets, transforms
import matplotlib as mpl
import copy
import sys
import matplotlib as mpl
from sklearn.cluster import KMeans
from scipy.ndimage import gaussian_filter, binary_opening, binary_closing
import argparse

# Get the parent directory of the current directory
project_root = os.path.abspath("..")
if project_root not in sys.path:
    sys.path.append(project_root)
    
from ddpm.unet import Unet
from ddpm.diffusion_proc import Diffusion, linear_noise_schedule
from ddpm.dataset import CondSeqImageDataset

###### LOAD CMAPS ############
cmap = mpl.colors.ListedColormap(['orange','yellow', 'green', 'black'])
cmap.set_over('0.25')
cmap.set_under('0.75')
bounds =  [0.0,0.1188,0.2798,0.7,1.1] #[0.0,0.2488,0.3098,0.7,1.1] # [1.0, 2.02, 2.27, 3.5, 5.1]
norm = mpl.colors.BoundaryNorm(bounds, cmap.N)
binary_cmap = mpl.colors.ListedColormap(['blue', 'black', 'red'])
under_cmap = mpl.colors.ListedColormap(['blue', 'black'])
over_cmap = mpl.colors.ListedColormap(['black', 'red'])

def load_background(BACKGROUND_PATH, cmap, norm):
    with open(BACKGROUND_PATH, 'rb') as f:
        background_img = pickle.load(f)

    plt.imshow(background_img, cmap=cmap, norm=norm)
    plt.colorbar()
    plt.axis("off")
    plt.show()
    
    return background_img

def generate_predictions(t_0, diff_model, transform, diff_client, num_preds=4, device="cuda"):
    input_img = t_0

    input_img = input_img.to(device).float().unsqueeze(0)
    image_size = input_img.shape[-1]
    channels = input_img.shape[1]

    input_arr = [input_img]
    outputs = []
    for i in range(num_preds):
        generated_images = diff_client.sample(
            model=diff_model,
            image_size=image_size,
            cond_img=input_arr[-1],
            batch_size=1,  # Set the desired batch size
            channels=channels,
            w=2,
            clip_denoised=True
        )
        otpt = generated_images[-1].squeeze().squeeze()
        otpt = np.where(otpt <= 0.5, 0, otpt)
        otpt = np.where(otpt > 0.5, 1, otpt)
        
        outputs.append(otpt)
        
        
        next_input = transform(otpt).to(device).float().unsqueeze(0)
        input_arr.append(next_input)

    return outputs

def compute_pixel_mismatch(image1, image2):
    """
    Compute the pixel mismatch between two binary images and return a binary image representing the mismatch.

    Parameters:
    - image1: numpy array representing the first binary image.
    - image2: numpy array representing the second binary image.

    Returns:
    - mismatch_image: A binary image where mismatched pixels are 1 and matching pixels are 0.
    - mismatch_percentage: The percentage of mismatched pixels.
    - over_under_estimate: An image where:
        - 1 represents overestimated pixels (present in image1 but not in image2),
        - -1 represents underestimated pixels (present in image2 but not in image1),
        - 0 represents matching pixels.
    """
    # Convert torch tensors to numpy arrays if needed
    if isinstance(image1, torch.Tensor):
        image1 = image1.numpy()
    if isinstance(image2, torch.Tensor):
        image2 = image2.numpy()
        
    # Ensure both images have the same shape
    if image1.shape != image2.shape:
        raise ValueError("The two images must have the same dimensions.")
    
    # Compute the mismatch by comparing the images
    mismatch_image = (image1 != image2).astype(int)  # Convert boolean array to integer array
    
    # Calculate the percentage of mismatched pixels
    total_pixels = image1.size
    mismatched_pixels = np.sum(mismatch_image)
    mismatch_percentage = round((mismatched_pixels / total_pixels) * 100, 3)
    
    # Create over_under_estimate image
    over_under_estimate = np.zeros_like(image1, dtype=int)
    
    # Label overestimated pixels as 1 (present in image1 but not in image2)
    over_under_estimate[(image1 == 1) & (image2 == 0)] = 1
    
    # Label underestimated pixels as -1 (present in image2 but not in image1)
    over_under_estimate[(image1 == 0) & (image2 == 1)] = -1

    # Calculate the percentages
    overestimated_pixels = np.sum(over_under_estimate == 1)
    underestimated_pixels = np.sum(over_under_estimate == -1)
    overestimate_percentage = round((overestimated_pixels / total_pixels) * 100, 3)
    underestimate_percentage = round((underestimated_pixels / total_pixels) * 100, 3)
    
    return mismatch_image, over_under_estimate, overestimate_percentage, underestimate_percentage, mismatch_percentage

def plot_images_in_row(images, background_img, type_output="simulation", mismatch_stat=None):
    """
    Function to plot 4 images in a single row, side by side.

    Parameters:
    - images: List or array of 4 images to be plotted.
    - background_img: The background image to combine with the images.
    - cmap: The colormap for the main images.
    - norm: The normalization for the colormap.
    - binary_cmap: The colormap for binary mismatch images.
    - under_cmap: The colormap for underestimation mismatch images.
    - over_cmap: The colormap for overestimation mismatch images.
    - type_output: The type of output to determine the titles and display ("simulation", "predicted (diffusion)", or "mismatch (diffusion)").
    - mismatch_stat: List of mismatch statistics if type_output is "mismatch (diffusion)".
    """

    # Ensure the input images are in a list or array and exactly 4 images
    assert len(images) == 4, "The images array must contain exactly 4 images."
    
    # Create a figure with a grid layout with 1 row and 4 columns
    fig, axes = plt.subplots(1, 4, figsize=(12, 6))

    # Plot the 4 images
    for i in range(4):
        img = images[i]
        
        if type_output in ["simulation", "predicted (diffusion)"]:
            combined_img = np.where(img == 1, 1, background_img)
            axes[i].imshow(combined_img, cmap=cmap, interpolation="none", norm=norm)
            
            if type_output == "simulation":
                axes[i].set_title(f'Simulated t = {(i+1) * 5}')
            elif type_output == "predicted (diffusion)":
                axes[i].set_title(f'Diffusion Predicted t = {(i+1) * 5}')
                
        elif type_output == "mismatch (diffusion)" and mismatch_stat is not None:
            over_perc, under_perc, total_mismatch = mismatch_stat[i]
            print(np.unique(img))
            
            if len(np.unique(img)) == 3:
                axes[i].imshow(img, cmap=binary_cmap)
            elif np.array_equal(np.unique(img), [-1, 0]):
                axes[i].imshow(img, cmap=under_cmap)
            elif np.array_equal(np.unique(img), [0, 1]):
                axes[i].imshow(img, cmap=over_cmap)
                
            axes[i].set_title(f'Mismatch')
        
        axes[i].axis('off')

    # Adjust the layout
    plt.tight_layout()
    plt.show() 
    
def main(args):
    
    ######### LOAD BACKGROUND IMAGE ##############
    background_img = load_background(args.background_path, cmap, norm)
   
    ######### LOAD MODEL ###############
    device = "cuda" if torch.cuda.is_available() else "cpu"
    diff_model =  Unet(
        in_ch=1,
        cond_ch=1,
        model_ch=96,
        output_ch=1,
        channel_mult=(1, 2, 2),
        attn_res=[],
    )
    diff_model.load_state_dict(torch.load(args.model_path))
    diff_model.to(device)
    
    ######### LOAD DATASET ############
    transform = transforms.Compose([
    transforms.ToTensor()
    ])
    simulated_data = CondSeqImageDataset(args.data_dir, transform=transform)
    
    ######### LOAD DIFFUSION HANDLER ###########
    diffusion = Diffusion(timesteps=500, noise_schedule='linear')
    
    sequence_size = 16
    for batch in range(10):
        start_index = batch * sequence_size
        end_index = start_index + sequence_size
        
        # Load the batch of data
        data = [simulated_data[i] for i in range(start_index, end_index)]
        input_img = data[0][1]  # Select the input image

        # Generate predictions using the diffusion model
        outputs_diffusion = generate_predictions(input_img, diff_model, transform, diffusion, device=device)
        # all_preds_diffusion.append(outputs_diffusion)

        # Plot diffusion model predictions
        plot_images_in_row(outputs_diffusion, background_img, type_output="predicted (diffusion)")

        # Prepare ground truth images
        ground_truth = []
        for i in range(0, len(data), 5):  # Adjust step as needed
            ground_truth.append(data[i][0].squeeze())
        # all_gts.append(ground_truth)
        
        # Plot ground truth images
        plot_images_in_row(ground_truth, background_img, type_output="simulation")

        # Compute mismatches for diffusion model predictions
        mismatch_diffusion = []
        mismatch_percentages_diffusion = []
        for i in range(4):
            predicted = outputs_diffusion[i]
            simulated = ground_truth[i]
            mismatch_image, over_under_estimate, over_percent, under_percent, mismatch_percentage = compute_pixel_mismatch(predicted, simulated)
            mismatch_diffusion.append(over_under_estimate)
            mismatch_percentages_diffusion.append((over_percent, under_percent, mismatch_percentage))
            print(f"Diffusion - Overestimate: {over_percent}% | Underestimate: {under_percent}%, Total Mismatch: {mismatch_percentage}%")
        
        # all_mismatches_diffusion.append(mismatch_diffusion)
        plot_images_in_row(mismatch_diffusion, type_output="mismatch (diffusion)", mismatch_stat=mismatch_percentages_diffusion)
        
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the simulation and diffusion process.")
    
    parser.add_argument('--data_dir', type=str, required=True, help="Path to the evaluation data directory.")
    parser.add_argument('--model_path', type=str, required=True, help="Path to the diffusion model checkpoint.")
    parser.add_argument('--baseline_path', type=str, required=True, help="Path to the baseline model checkpoint.")
    parser.add_argument('--background_path', type=str, required=True, help="Path to the background image file.")

    args = parser.parse_args()
    
    main(args)
    
    
    
    