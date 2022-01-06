import sys
import os
import shutil
import random

#SRC_VIDEO_DIR = r'E:\workspace\pro\smoke_keypoint\data\search_20220104_cls\fenlei\非抽烟-背景'
SRC_VIDEO_DIR = r'E:\workspace\pro\smoke_keypoint\data\test\shanghai\error'
DST_VIDEO_DIR = r'E:\workspace\pro\smoke_keypoint\data\test\search_20220104_negative'

if not os.path.exists(DST_VIDEO_DIR):
    os.makedirs(DST_VIDEO_DIR)
    
vs = os.listdir(SRC_VIDEO_DIR)

random.shuffle(vs)
vs = vs[:600]
for v in vs:
    shutil.copy(os.path.join(SRC_VIDEO_DIR, v), os.path.join(DST_VIDEO_DIR, v))