import sys
import os
import numpy as np

import cv2
import json

ANNS_FILE_PATH = r'H:\pro\smoke_keypoint\data\smoke_keypoint_mpii\annotations\train.json'
OUT_ANNS_PATH = r'H:\pro\smoke_keypoint\data\smoke_keypoint_mpii\annotations\new_center_train.json'

IMG_ROOT_DIR = r'H:\pro\smoke_keypoint\data\smoke_keypoint_mpii\images'

# load old json info
with open(ANNS_FILE_PATH, 'r') as fp:
    anns = json.load(fp)

# load facerect info
facerect_file_path = r'H:\pro\smoke_keypoint\data\smoke_keypoint_mpii\annotations\facerect.npy'
rect_infos = np.load(facerect_file_path, allow_pickle=True).item()

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

new_ann = []
for ann in anns:
    # get image
    img = ann['image']

    # check if the image valid
    if img in rect_infos and rect_infos[img] != None:
        # read facerect
        facerect = rect_infos[img]
        # read image
        image = cv2.imread(os.path.join(IMG_ROOT_DIR, img))
        cx,cy = get_center(image, facerect)
        # update
        ann['center'] = [cx,cy]

        new_ann.append(ann)

print("Old {}, New {}".format(len(anns), len(new_ann)))

with open(OUT_ANNS_PATH, 'w') as fp:
    json.dump(new_ann, fp)


