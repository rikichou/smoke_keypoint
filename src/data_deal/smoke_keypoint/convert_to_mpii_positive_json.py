import os

import json
import glob
from pathlib import Path
import shutil
import numpy as np
import cv2

level = 3
prefix = 'positive'
SRC_ROOT_IMG_DIR = 'H:\pro\smoke_keypoint\data\smoke_keypoint_relabel'
OUT_ROOT_DIR =  'H:\pro\smoke_keypoint\data\smoke_keypoint_mpii_relabel'
OUT_ROOT_IMG_DIR = os.path.join(OUT_ROOT_DIR, r'images\positive')
if not os.path.exists(OUT_ROOT_IMG_DIR):
    os.makedirs(OUT_ROOT_IMG_DIR)
OUT_ROOT_ANNS_DIR = os.path.join(OUT_ROOT_DIR, 'annotations')
if not os.path.exists(OUT_ROOT_ANNS_DIR):
    os.makedirs(OUT_ROOT_ANNS_DIR)
OUT_ANNS_PATH = os.path.join(OUT_ROOT_ANNS_DIR, 'positive.json')

def is_image(name):
    img_ext = ['.jpg', '.jpeg', '.png', '.bmp', '.tif']
    prefix,ext = os.path.splitext(name)
    ext = ext.lower()
    if ext in img_ext:
        return True
    else:
        return False

files = glob.glob(SRC_ROOT_IMG_DIR + str(Path('/*' * level)))

facerect_file_path = r'H:\pro\smoke_keypoint\data\smoke_keypoint_mpii\images\positive\facerect.npy'
rect_infos = np.load(facerect_file_path, allow_pickle=True).item()

def get_center(image, facerect):
    sx, sy, ex, ey = facerect
    h, w, c = image.shape

    res = 200
    cx = sx + (ex-sx)/2
    cy = ey - (ey-sy)/3

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
        try:
            ann = json.load(fp)
        except:
            print("Failed to Load json file ", img_json_path)
            continue

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
    out_json_info['image'] = prefix + '/' + image_name

    # scale
    out_json_info['scale'] = 1.0

    # center
    if image_name not in rect_infos:
        continue
    facerect = rect_infos[image_name]
    if facerect == None:
        continue
    image = cv2.imread(img)
    cx,cy = get_center(image, facerect)
    out_json_info['center'] = [cx, cy]

    # show image
    # sx = int(cx) - 200
    # ex = int(cx) + 200
    # sy = int(cy) - 200
    # ey = int(cy) + 200
    # cv2.rectangle(image, (sx,sy), (ex, ey), (0,255,0), 2)
    # cv2.rectangle(image, (facerect[0], facerect[1]), (facerect[2], facerect[3]), (255, 0, 0), 2)
    # cv2.imshow('pose', image)
    # if cv2.waitKey(0) == ord('q'):
    #     break

    # point type
    try:
        out_json_info['point_type'] = 'with_hand' if ann['shapes'][0]['point_type'][0] == 2 else "no_hand"
    except:
        print("have no point type ", img_json_path)
        continue

    out_mpii_json.append(out_json_info)

    # copy image to dst dir
    out_img_path = os.path.join(OUT_ROOT_IMG_DIR, image_name)

    if not os.path.exists(out_img_path):
        shutil.copy(img, out_img_path)

    if idx%1000 == 0:
        print("{}/{}".format(idx, len(files)))

cv2.destroyAllWindows()

with open(OUT_ANNS_PATH, 'w') as fp:
    json.dump(out_mpii_json, fp)

















