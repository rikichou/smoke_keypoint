import os
import sys
import glob
import json

VIDEO_ROOT_DIR = r'Y:\pro\smoke_keypoints\smoke_keypoint\data\collect_chepai'
OUT_NAME_PATH = r'E:\workspace\pro\smoke_keypoint\data\selected\name.json'

vs = glob.glob(VIDEO_ROOT_DIR+'\\*\\*')
names = [os.path.basename(tmp) for tmp in vs]

names = [v.rsplit('.', maxsplit=1)[0] for v in names]

#names = [str1[:str1.find('.264')+4] for str1 in names]

info = {'names':names}
with open(OUT_NAME_PATH, 'w') as fp:
    json.dump(info, fp)


