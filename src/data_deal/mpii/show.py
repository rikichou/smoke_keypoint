import sys
import os

import cv2
import json

ANNS_FILE_PATH = '/Volumes/10.20.132.160-1/pro/smoke_keypoint/data/mpii/untar/annotations/mpii_val.json'
IMG_ROOT_DIR = '/Volumes/10.20.132.160-1/pro/smoke_keypoint/data/mpii/untar/images'

with open(ANNS_FILE_PATH, 'r') as fp:
    anns = json.load(fp)

for ann in anns:
    print(ann)
    # get image
    img = ann['image']
    img_path = os.path.join(IMG_ROOT_DIR, img)
    img = cv2.imread(img_path)

    # write points
    points = ann['joints']
    for p in points:
        x,y = p
        cv2.circle(img, (int(x),int(y)), 1, (0,0,255), 2)
    cx,cy = ann['center']
    cv2.circle(img, (int(cx), int(cy)), 1, (0, 255, 255), 5)

    # show
    cv2.imshow('pose', img)
    if cv2.waitKey(0) == ord('q'):
        break

cv2.destroyAllWindows()


