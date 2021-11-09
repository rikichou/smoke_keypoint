import sys
import os

import cv2
import json

ANNS_DIR = r'H:\pro\smoke_keypoint\data\smoke_keypoint_relabel\20210728\4'

fs = os.listdir(ANNS_DIR)

for f in fs:
    if '.json' not in f:
        continue

    anns_fp = os.path.join(ANNS_DIR, f)
    with open(anns_fp, 'r') as fp:
        ann = json.load(fp)

    # get image
    img_path = os.path.join(ANNS_DIR, ann['imagePath'])
    print(img_path)
    img = cv2.imread(img_path)

    # write points
    points = ann['shapes'][0]['points']
    color_map = {0:(255,0,0), 1:(0,255,0), 2:(0,0,255)}
    for idx,p in enumerate(points):
        x,y = p
        cv2.circle(img, (int(x),int(y)), 1, color_map[idx], 2)

    # show
    cv2.imshow('keypoints', img)
    if cv2.waitKey(0) == ord('q'):
        break

cv2.destroyAllWindows()


