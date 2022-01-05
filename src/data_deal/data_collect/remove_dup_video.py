import os
import glob
import shutil

VIDEO_ROOT_DIR = r'E:\workspace\pro\smoke_keypoint\data\search'

all_264 = glob.glob(os.path.join(VIDEO_ROOT_DIR, '*.264'))

count = 0
for idx,v in enumerate(all_264):
    vpath = os.path.splitext(v)[0] + '.avi'
    if os.path.exists(vpath):
        os.remove(vpath)
        count += 1

    if idx%1000 == 0:
        print("{}/{}".format(idx, len(all_264)))

print("Total remove ", count)