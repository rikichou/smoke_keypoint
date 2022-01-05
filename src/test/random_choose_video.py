import sys
import os
import shutil
import random

SRC_VIDEO_DIR = r'E:\workspace\pro\smoke_keypoint\data\search_20220104_cls\fenlei\抽烟'
DST_VIDEO_DIR = r'E:\workspace\pro\smoke_keypoint\data\test\search_20220104_smoke'

if not os.path.exists(DST_VIDEO_DIR):
    os.makedirs(DST_VIDEO_DIR)
    
vs = os.listdir(SRC_VIDEO_DIR)

random.shuffle(vs)
vs = vs[:110]
for v in vs:
    shutil.copy(os.path.join(SRC_VIDEO_DIR, v), os.path.join(DST_VIDEO_DIR, v))