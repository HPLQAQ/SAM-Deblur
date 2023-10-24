import os
import shutil

def move_and_rename_files(src_dir, tar_dir):
    # 确保目标文件夹存在
    if not os.path.exists(tar_dir):
        os.makedirs(tar_dir)

    # 列出源目录中的所有文件和文件夹
    for filename in os.listdir(src_dir):
        if filename.startswith("mask_"):
            src_path = os.path.join(src_dir, filename)
            new_filename = filename.replace("mask_", "", 1)
            tar_path = os.path.join(tar_dir, new_filename)

            # 移动并重命名文件
            shutil.move(src_path, tar_path)
            print(f"Moved {src_path} to {tar_path}")

# 源目录和目标目录（根据你的实际路径进行修改）
src_dir = "/home/henryli/project/NAFNet/datasets/ReLoBlur/test/input"
tar_dir = "/home/henryli/project/NAFNet/datasets/ReLoBlur/test/input_mask"

move_and_rename_files(src_dir, tar_dir)
