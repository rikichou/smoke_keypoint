import shutil
import os

VIDEO_ROOT_DIR = r'H:\pro\smoke_keypoint\data\test\collect\videos\collect\negative'
dirlist = os.listdir(VIDEO_ROOT_DIR)

count = 0
for idx,v in enumerate(dirlist):
    dirpath = os.path.join(VIDEO_ROOT_DIR, v)
    if os.path.isdir(dirpath) and '.264' == v[-4:]:
        shutil.rmtree(dirpath)
        count += 1

print("Total removed {} videos".format(count))
