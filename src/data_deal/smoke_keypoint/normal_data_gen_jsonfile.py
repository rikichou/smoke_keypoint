import os
import numpy as np
import json
import glob
from pathlib import Path
import shutil
import cv2

level = 1

SRC_ROOT_IMG_DIR = '/zhourui/workspace/pro/smoke_keypoint/data/smoke_keypoint_mpii/train/negative_images'

OUT_ROOT_ANNS_DIR = '/zhourui/workspace/pro/smoke_keypoint/data/smoke_keypoint_mpii/annotations'
if not os.path.exists(OUT_ROOT_ANNS_DIR):
    os.makedirs(OUT_ROOT_ANNS_DIR)
OUT_ANNS_PATH = os.path.join(OUT_ROOT_ANNS_DIR, 'fatigue_negative.json')

# load facerect info
facerect_file_path = os.path.join(SRC_ROOT_IMG_DIR, 'facerect.npy')
if not os.path.exists(facerect_file_path):
    print("Error !", facerect_file_path)
rect_infos = np.load(facerect_file_path, allow_pickle=True).item()

def is_image(name):
    img_ext = ['.jpg', '.jpeg', '.png', '.bmp', '.tif']
    prefix,ext = os.path.splitext(name)
    ext = ext.lower()
    if ext in img_ext:
        return True
    else:
        return False

def get_center(image, facerect):
    sx, sy, ex, ey = facerect
    h, w, c = image.shape

    res = 200
    cx = sx + (ex-sx)/2
    cy = ey

    osx = cx - res
    osy = cy - res
    oex = cx + res
    oey = cy + res

    sx = int(max(0, osx))
    sy = int(max(0, osy))
    ex = int(min(w - 1, oex))
    ey = int(min(h - 1, oey))

    cx = sx + (ex-sx)/2
    cy = sy + (ey-sy)/2

    return int(cx), int(cy)

files = glob.glob(SRC_ROOT_IMG_DIR + str(Path('/*' * level)))

out_mpii_json = []
for idx,f in enumerate(files):
    if not is_image(f):
        continue
    img = f

    out_json_info = {}
    # check if joints is visible
    out_json_info['joints_vis'] = [0.49,0.49,0.49]

    # all joints
    out_json_info['joints'] = [[-1, -1],[-1, -1],[-1, -1]]

    # image name
    _,dname,imgname = img.rsplit(img, maxsplit=2)
    out_json_info['image'] = os.path.join(dname, imgname)

    # scale
    out_json_info['scale'] = 1.0

    # center(according to the face rect)
    if imgname not in rect_infos:
        print("Can not found image face rect ", imgname)
        continue
    facerect = rect_infos[imgname]

    image = cv2.imread(f)
    cx, cy = get_center(image, facerect)
    # update
    out_json_info['center'] = [cx, cy]

    #
    out_json_info['point_type'] = 'negative'

    out_mpii_json.append(out_json_info)

    if idx%1000 == 0:
        print("{}/{}".format(idx, len(files)))

with open(OUT_ANNS_PATH, 'w') as fp:
    json.dump(out_mpii_json, fp)

















