import os

import json
import glob
from pathlib import Path
import shutil

level = 3

SRC_ROOT_IMG_DIR = '/Volumes/10.20.132.160-1/pro/smoke_keypoint/data/smoke_keypoint'
OUT_ROOT_DIR =  '/Volumes/10.20.132.160-1/pro/smoke_keypoint/data/smoke_keypoint_mpii'
OUT_ROOT_IMG_DIR = os.path.join(OUT_ROOT_DIR, 'images')
if not os.path.exists(OUT_ROOT_IMG_DIR):
    os.makedirs(OUT_ROOT_IMG_DIR)
OUT_ROOT_ANNS_DIR = os.path.join(OUT_ROOT_DIR, 'annotations')
if not os.path.exists(OUT_ROOT_ANNS_DIR):
    os.makedirs(OUT_ROOT_ANNS_DIR)
OUT_ANNS_PATH = os.path.join(OUT_ROOT_ANNS_DIR, 'train.json')

def is_image(name):
    img_ext = ['.jpg', '.jpeg', '.png', '.bmp', '.tif']
    prefix,ext = os.path.splitext(name)
    ext = ext.lower()
    if ext in img_ext:
        return True
    else:
        return False

files = glob.glob(SRC_ROOT_IMG_DIR + str(Path('/*' * level)))

out_mpii_json = []
for idx,f in enumerate(files):
    if not is_image(f):
        continue
    img = f
    out_json_info = {}
    # check if json file is exist
    img_json_path = os.path.splitext(f)[0]+'.json'
    if not os.path.exists(img_json_path):
        print("Can not found json ", img_json_path)
        continue

    # read all json info
    with open(img_json_path, 'r') as fp:
        ann = json.load(fp)

    # to mpii format

    # check if joints is visible
    out_json_info['joints_vis'] = [1,1,1]

    # all joints
    out_json_info['joints'] = ann['shapes'][0]['points']

    # image name
    image_name = os.path.basename(img)
    if image_name != ann['imagePath']:
        print("image name {}, imagepath {}".format(image_name, ann['imagePath']))
        continue
    out_json_info['image'] = image_name

    # scale
    out_json_info['scale'] = 1.0

    # center
    out_json_info['center'] = ann['shapes'][0]['points'][1]

    out_mpii_json.append(out_json_info)

    # copy image to dst dir
    out_img_path = os.path.join(OUT_ROOT_IMG_DIR, image_name)
    shutil.copy(img, out_img_path)

    if idx%1000 == 0:
        print("{}/{}".format(idx, len(files)))

with open(OUT_ANNS_PATH, 'w') as fp:
    json.dump(out_mpii_json, fp)

















