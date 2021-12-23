import sys
import os
import random
import cv2
import json

ANNS_FILE_PATH = r'H:\pro\smoke_keypoint\data\smoke_keypoint_mpii_relabel\annotations\valid_positive.json'
IMG_ROOT_DIR = r'H:\pro\smoke_keypoint\data\smoke_keypoint_mpii_relabel\images'

with open(ANNS_FILE_PATH, 'r') as fp:
    anns = json.load(fp)

random.shuffle(anns)
for ann in anns:
    # get image
    img = ann['image']
    # if '0000000000000000-190513-035851-035915-000003290330200.jpg' not in img:
    #     continue
    img_path = os.path.join(IMG_ROOT_DIR, img)
    img = cv2.imread(img_path)

    # write points
    points = ann['joints']
    color_map = {0: (255, 0, 0), 1: (0, 255, 0), 2: (0, 0, 255)}
    for idx,p in enumerate(points):
        x,y = p
        cv2.circle(img, (int(x),int(y)), 1, color_map[idx], 2)
    cx,cy = ann['center']
    cv2.circle(img, (int(cx), int(cy)), 1, (0, 255, 255), 5)

    sx = int(cx) - 200
    ex = int(cx) + 200
    sy = int(cy) - 200
    ey = int(cy) + 200
    cv2.rectangle(img, (sx,sy), (ex, ey), (0,255,0), 1)

    # show
    cv2.imshow('pose', img)
    if cv2.waitKey(0) == ord('q'):
        break

cv2.destroyAllWindows()


