import os
import shutil

SRC_V_ROOT_DIR = r'H:\pro\smoke_keypoint\data\video\test_OUT'
OUT_V_ROOT_DIR = r'H:\pro\smoke_keypoint\data\video\test'

num_perf = 3

dirs = os.listdir(SRC_V_ROOT_DIR)

for i in range(len(dirs)):
    folder_idx = i%num_perf
    
    out_dir = os.path.join(OUT_V_ROOT_DIR, str(folder_idx))
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
        
    shutil.move(os.path.join(SRC_V_ROOT_DIR, dirs[i]), out_dir)

