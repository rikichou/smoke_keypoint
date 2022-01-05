import os
import math
import shutil
import numpy as np

split_n = 6
SRC_ROOT_DIR = r'H:\pro\smoke_keypoint\data\test\collect\videos\collect\negative_together'
OUT_ROOT_DIR = r'H:\pro\smoke_keypoint\data\test\collect\videos\collect\split'
if not os.path.exists(OUT_ROOT_DIR):
    os.makedirs(OUT_ROOT_DIR)

imgs = np.array(os.listdir(SRC_ROOT_DIR))
files_split = np.array_split(imgs, split_n)

for i in range(split_n):
    out_dir = os.path.join(OUT_ROOT_DIR, str(i))
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    for img in files_split[i]:
        src_img_path = os.path.join(SRC_ROOT_DIR, img)
        out_img_path = os.path.join(out_dir, img)
        if not os.path.exists(out_img_path):
            shutil.copy(src_img_path, out_img_path)

    print(i)