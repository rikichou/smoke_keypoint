import sys
import os
import random
import cv2
import json

ANNS_FILE_PATH = r'H:\pro\smoke_keypoint\data\smoke_keypoint_mpii_relabel\annotations\val.json'
IMG_ROOT_DIR = r'H:\pro\smoke_keypoint\data\smoke_keypoint_mpii_relabel\images'
OUT_ANNS_FILE_PATH = r'H:\pro\smoke_keypoint\data\smoke_keypoint_mpii_relabel\annotations\val_new.json'

with open(ANNS_FILE_PATH, 'r') as fp:
    anns = json.load(fp)

count = 0
out_mpii_json = []
for ann in anns:
    # get image
    img = ann['image']
    # if '0000000000000000-190513-035851-035915-000003290330200.jpg' not in img:
    #     continue
    img_path = os.path.join(IMG_ROOT_DIR, img)
    if not os.path.exists(img_path):
        print(img_path)
        count += 1
    else:
        out_mpii_json.append(ann)

print(count)

if count>0:
    with open(OUT_ANNS_FILE_PATH, 'w') as fp:
        json.dump(out_mpii_json, fp)

print("Total {}, Invalid {} valid {}".format(len(anns), count, len(out_mpii_json)))


