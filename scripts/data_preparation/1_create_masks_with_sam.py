# ------------------------------------------------------------------------
# Copyright (c) 2023 Siwei Li.
# ------------------------------------------------------------------------
import numpy as np
import cv2
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator
import os

# Update with your actual paths
src_dir_list = ['./input']
tar_dir_list = ['./input_mask']

#your segement anything model path
sam_checkpoint = '/home/henryli/project/segment-anything/model/sam_vit_h.pth'     

device = "cuda"
model_type = "default"

sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
sam.to(device=device)
sam.eval()

def draw_segmask(masks):
    # Assuming the first mask's shape represents the shape of all masks
    segmask = np.zeros_like(masks[0]['segmentation'], dtype=np.float32)
    
    # Enumerate through masks to get id and segmentation data
    for idx, mask_dict in enumerate(masks):
        segmentation = mask_dict['segmentation']
        
        # Convert segmentation to a binary matrix (segmentation result of 1)
        binary_area = (segmentation > 0).astype(np.float32)
        
        # Update segmask based on the conditions provided in the pseudocode
        if idx < 127:
            segmask += binary_area * 2 * (idx + 1)
        else:
            segmask += binary_area * (2 * (255 - idx) - 1)
    
    # Convert segmask values to the range [0, 255]
    segmask = np.clip(segmask, 0, 255).astype(np.uint8)
    
    return segmask

def get_mask(image):
    mask_generator = SamAutomaticMaskGenerator(sam)
    masks = mask_generator.generate(image)

    segmask = draw_segmask(masks)
    return segmask

def process_directory(src_dir, tar_dir):
    """
    Process a directory and save generated masks to the target directory.
    Only process the images in the first level of the given directory (non-recursive).
    """
    filenames = [f for f in os.listdir(src_dir) if os.path.isfile(os.path.join(src_dir, f))]
    
    for filename in filenames:
        # Check if the file is a PNG and doesn't have the mask_ prefix
        if filename.endswith('.png') and not filename.startswith('mask_'):
            
            # Check if the mask already exists in the target directory
            mask_path = os.path.join(tar_dir, filename)
            if os.path.exists(mask_path):
                print(f"Mask for {filename} already exists. Skipping.")
                continue  # Skip this iteration and move on to the next file
            
            # Read the image
            img_path = os.path.join(src_dir, filename)
            image = cv2.imread(img_path)
            
            # Generate mask for the image
            mask = get_mask(image)
            
            # Save the mask in the target directory without changing the filename
            cv2.imwrite(mask_path, mask)

def process_directories(src_dir_list, tar_dir_list):
    """
    Process a list of source and target directories.
    """
    if len(src_dir_list) != len(tar_dir_list):
        print("Error: The number of source and target directories must be the same.")
        return
    
    for src_dir, tar_dir in zip(src_dir_list, tar_dir_list):
        # Create target directory if it doesn't exist
        if not os.path.exists(tar_dir):
            os.makedirs(tar_dir)
            
        process_directory(src_dir, tar_dir)

process_directories(src_dir_list, tar_dir_list)