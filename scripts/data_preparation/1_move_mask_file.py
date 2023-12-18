# ------------------------------------------------------------------------
# Copyright (c) 2023 Siwei Li.
# ------------------------------------------------------------------------
import os
import shutil

# source directory and target directory (modify according to your actual path)
src_dir = "../../datasets/GoPro/train/input"
tar_dir = "../../datasets/GoPro/train/input_mask"


def move_and_rename_files(src_dir, tar_dir):
    # ensure the target folder exists
    if not os.path.exists(tar_dir):
        os.makedirs(tar_dir)

    # list all the files and folders in the source directory
    for filename in os.listdir(src_dir):
        if filename.startswith("mask_"):
            src_path = os.path.join(src_dir, filename)
            new_filename = filename.replace("mask_", "", 1)
            tar_path = os.path.join(tar_dir, new_filename)

            # move and rename files
            shutil.move(src_path, tar_path)
            print(f"Moved {src_path} to {tar_path}")

move_and_rename_files(src_dir, tar_dir)
