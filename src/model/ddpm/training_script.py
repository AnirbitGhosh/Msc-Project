import os
import math
from abc import abstractmethod

from PIL import Image
import requests
import numpy as np
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
from tqdm import tqdm
import matplotlib.pyplot as plt
from torchvision import transforms, datasets
from torch.utils.data import Dataset, DataLoader
import pickle
from PIL import Image
import cv2
import numpy as np
import matplotlib as mpl
from sklearn.cluster import KMeans
from scipy.ndimage import gaussian_filter, binary_opening, binary_closing
import copy
import sys
import argparse  # Import argparse

# Get the parent directory of the current directory
project_root = os.path.abspath("..")
if project_root not in sys.path:
    sys.path.append(project_root)
    
from ddpm.unet import Unet, UpSample, DownSample, time_embedding
from ddpm.diffusion_proc import Diffusion, linear_noise_schedule
from ddpm.blocks import ResBlock, AttentionBlock, TimestepBlock, group_norm_layer
from ddpm.dataset import CondSeqImageDataset


def train(model, train_loader, diffusion_client, epochs=20, p_uncound=0.2):
    # train
    p_uncound = p_uncound
    len_data = len(train_loader)
    time_end = time.time()

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    diff_losses = []
    for epoch in range(epochs):
        epoch_diff_losses = []
        for step, (images, cond_images) in enumerate(train_loader):     
            time_start = time_end
            
            optimizer.zero_grad()
            
            batch_size = images.shape[0]
            images = images.to(device).float()
            cond_images = cond_images.to(device).float()
            
            # random generate mask
            z_uncound = torch.rand(batch_size)
            batch_mask = (z_uncound > p_uncound).int().to(device)
            
            # sample t uniformally for every example in the batch
            t = torch.randint(0, timesteps, (batch_size,), device=device).long()
            loss = diffusion_client.train_losses(model, images, t, cond_images, batch_mask)

            epoch_diff_losses.append(loss.item())
            
            if step % 100 == 0:
                time_end = time.time()
                print("Epoch {}/{}\t Step {}/{}\t Loss {:.4f}\t Time {:.2f}".format(epoch+1, epochs, step+1, len_data, loss.item(), time_end-time_start))
                
            loss.backward()
            optimizer.step()
            
        diff_losses.append(np.array(epoch_diff_losses).mean())
        
        return diff_losses
    
def plot_loss_curve(losses, title, epochs=20):
    xticks = list(range(epochs))
    plt.figure(figsize=(10, 5))
    plt.plot(losses, label='Diffusion Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title(title)
    plt.legend()
    plt.xticks(xticks)
    plt.show()
    
def save_model(model, path):
    torch.save(model.state_dict(), path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a DDPM model.")
    parser.add_argument('--data_dir', type=str, required=True, help="Path to the dataset directory")
    parser.add_argument('--result_dir', type=str, required=True, help="Path to the result directory")
    parser.add_argument('--ckpt_dir', type=str, required=True, help="Path to the checkpoint directory")

    args = parser.parse_args()

    batch_size = 32
    timesteps = 500

    transform = transforms.Compose([
        transforms.ToTensor(),
    ])

    # Use the paths provided as arguments
    DATA_DIR = args.data_dir
    RESULT_DIR = args.result_dir
    CKPT_DIR = args.ckpt_dir

    # MNIST DATA
    dataset = CondSeqImageDataset(DATA_DIR, transform=transform)

    # Split dataset into training and testing
    train_size = int(0.9 * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])

    # Create DataLoader
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = Unet(
        in_ch=1,
        cond_ch=1,
        model_ch=96,
        output_ch=1,
        channel_mult=(1, 2, 2),
        attn_res=[],
    )
    model.to(device)
    
    diffusion_client = Diffusion(timesteps=timesteps, noise_schedule='sigmoid')
    
    training_losses = train(model, train_loader, diffusion_client)
    plot_loss_curve(training_losses, title="Training losses")
    
    save_model(model, os.path.join(CKPT_DIR, "ddpm_model.pt"))