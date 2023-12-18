# ------------------------------------------------------------------------
# Copyright (c) 2023 Siwei Li.
# ------------------------------------------------------------------------
import shutil
import os

# list_dir is the path to dir.txt
list_dir = '../../datasets/RealBlur/RealBlur_J_train_list.txt'

# target_dir is the directory you want to save the files
target_dir = '../../datasets/RealBlurJ/train'
src_dir = '../../datasets/RealBlur'

# ensure that the target directory exists
if not os.path.exists(os.path.join(target_dir, 'target')):
    os.makedirs(os.path.join(target_dir, 'target'))
if not os.path.exists(os.path.join(target_dir, 'input')):
    os.makedirs(os.path.join(target_dir, 'input'))

# read dir.txt
with open(list_dir, 'r') as f:
    lines = f.readlines()

# process each line
for line in lines:
    gt_path, blur_path = line.strip().split(' ')
    
    # generate new file name
    scene_name = gt_path.split('/')[1]
    file_name = gt_path.split('/')[-1].split('.')[0]
    new_file_name = f'RealBlur-J-{scene_name}-{file_name}.png'
    
    # copy target (Ground Truth) and input (Blurred) images to new directory
    shutil.copy(os.path.join(src_dir, gt_path), os.path.join(target_dir, 'target', new_file_name))
    shutil.copy(os.path.join(src_dir, blur_path), os.path.join(target_dir, 'input', new_file_name))
