import shutil
import os

# 假设 list_dir 是 dir.txt 文件的路径
list_dir = '/home/henryli/data/RealBlur_J_train_list.txt'

# 假设 target_dir 是你想要保存文件的目录
target_dir = '/home/henryli/data/RealBlurJ/train'
src_dir = '/home/henryli/data/'

# 确保目标目录存在
if not os.path.exists(os.path.join(target_dir, 'target')):
    os.makedirs(os.path.join(target_dir, 'target'))
if not os.path.exists(os.path.join(target_dir, 'input')):
    os.makedirs(os.path.join(target_dir, 'input'))

# 读取 dir.txt 文件
with open(list_dir, 'r') as f:
    lines = f.readlines()

# 处理每一行
for line in lines:
    gt_path, blur_path = line.strip().split(' ')
    
    # 生成新的文件名
    scene_name = gt_path.split('/')[1]
    file_name = gt_path.split('/')[-1].split('.')[0]
    new_file_name = f'RealBlur-J-{scene_name}-{file_name}.png'
    
    # 复制目标（Ground Truth）和输入（Blurred）图片到新的目录
    shutil.copy(os.path.join(src_dir, gt_path), os.path.join(target_dir, 'target', new_file_name))
    shutil.copy(os.path.join(src_dir, blur_path), os.path.join(target_dir, 'input', new_file_name))
