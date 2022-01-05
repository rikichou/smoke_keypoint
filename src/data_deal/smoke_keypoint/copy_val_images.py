import os

import json
import glob
import shutil

json_path = r'H:\pro\smoke_keypoint\data\smoke_keypoint_mpii_relabel\annotations\val.json'
ROOT_IMG_DIR = r'H:\pro\smoke_keypoint\data\smoke_keypoint_mpii_relabel\images'
OUT_ROOT_IMG_DIR = r'H:\pro\smoke_keypoint\data\val'
if not os.path.exists(OUT_ROOT_IMG_DIR):
    os.makedirs(OUT_ROOT_IMG_DIR)

with open(json_path, 'r') as fp:
    anns = json.load(fp)

for ann in anns:
    img_name = ann['image']
    out_name = os.path.basename(img_name)

    shutil.copy(os.path.join(ROOT_IMG_DIR, img_name), os.path.join(OUT_ROOT_IMG_DIR, out_name))


