import os
import shutil
import time

SRC_V_ROOT_DIR = r'H:\pro\smoke_keypoint\data\video\rawframes_split\2'
OUT_V_ROOT_DIR = r'H:\pro\smoke_keypoint\data\video\rawframes_s'

num_perf = 1000

# 3,7,

dirs = os.listdir(SRC_V_ROOT_DIR)

count = 0
for i in range(len(dirs)):
    folder_idx = int(count/num_perf) + 8
    
    # time clean
    path = os.path.join(SRC_V_ROOT_DIR, dirs[i])
    file_ct_time = time.localtime(os.path.getctime(path))
    if file_ct_time.tm_hour >= 16:
        continue
    
    out_dir = os.path.join(OUT_V_ROOT_DIR, str(folder_idx))
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
        
    shutil.move(os.path.join(SRC_V_ROOT_DIR, dirs[i]), out_dir)
    
    count += 1

