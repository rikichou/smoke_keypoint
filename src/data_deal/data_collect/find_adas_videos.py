import sys
import os
import glob
import shutil

ROOT_VIDEO_DIR = r'H:\pro\smoke_keypoint\data\test\collect\videos\collect\videos_11181900'
fullpath_list = glob.glob(os.path.join(ROOT_VIDEO_DIR, '*'))

OUT_ROOT_DIR = r'H:\pro\smoke_keypoint\data\test\collect\videos\collect\videos_11181600_adas'
if not os.path.exists(OUT_ROOT_DIR):
    os.makedirs(OUT_ROOT_DIR)

count = 0
for idx,avi in enumerate(fullpath_list):
    if '01p015000000' in avi or '_CH1' in avi or 'channelNo0.' in avi or '01p015000000' in avi or '01p011000000' in avi or '01p211000000' in avi:
        dst_avi = os.path.join(OUT_ROOT_DIR, os.path.basename(avi))
        #shutil.copy(avi, dst_avi)
        os.remove(avi)
        count += 1

    if idx%1000 == 0:
        print("{}/{}".format(idx, len(fullpath_list)))
print("Del ", count)

