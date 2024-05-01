import numpy as np
import cv2
import pickle 
import argparse 
import os
import matplotlib.pyplot as plt 

def dir_path(string):
    if os.path.isdir(string):
        return string
    else:
        raise NotADirectoryError(string)
    
parser = argparse.ArgumentParser()
parser.add_argument('-d', '--data', type=dir_path, help='Pass data directory path using -d or --data flags')

# Perform grayscale conversion
def rgb_to_gray(rgb_images):
    gray_images = np.zeros((rgb_images.shape[0], rgb_images.shape[1], rgb_images.shape[2]), dtype=np.uint8)
    for i in range(rgb_images.shape[0]):
        gray_images[i] = cv2.cvtColor(rgb_images[i], cv2.COLOR_RGB2GRAY)
    return gray_images

# Binarize the grayscale images (convert to binary images)
def binarize_images(gray_images, threshold=128):
    normalized_images = gray_images / 255.0
    return normalized_images

if __name__ == "__main__":
    args = parser.parse_args()
    
    data_dir = args.data
    frame_dir = os.path.join(data_dir, "frames")
    bin_frame_dir = os.path.join(data_dir, "bin_frames")
    
    if not os.path.exists(bin_frame_dir):
        os.makedirs(bin_frame_dir)
    
    file_list = os.listdir(frame_dir)
    for file in file_list:
        with open(os.path.join(frame_dir, file), 'rb') as f:
            data = np.array(pickle.load(f))
        
        gray_images = rgb_to_gray(data)
        
        # Set a threshold to binarize the grayscale images
        threshold = 128
        binary_images = binarize_images(gray_images, threshold)
        
        out_name = os.path.join(bin_frame_dir, file)
        with open(out_name, 'wb') as f:
            pickle.dump(binary_images, f)
        
    
    
