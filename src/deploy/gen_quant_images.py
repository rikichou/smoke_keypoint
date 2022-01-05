import sys
import os
import json
import cv2
import numpy as np
import random

json_path = r'H:\pro\smoke_keypoint\data\smoke_keypoint_coco\annotations\val_coco.json'
img_root_dir = r'H:\pro\smoke_keypoint\data\smoke_keypoint_coco\images'

OUT_ROOT_DIR = r'E:\software\haisi\RuyiStudio-2.0.46\workspace\smoke_keypoints\smoke_keypoints\data\images'
if not os.path.exists(OUT_ROOT_DIR):
    os.makedirs(OUT_ROOT_DIR)

with open(json_path, 'r') as fp:
    infos = json.load(fp)

    images = infos['images']
    anns = infos['annotations']

    idx_list = list(range(len(images)))
    
    for idx in idx_list:
        info = images[idx]
        # read image
        img_path = os.path.join(img_root_dir, info['file_name'])
        image = cv2.imread(img_path)
        if image is None:
            print("Failed to open ", img_path)
            continue
        
        # show smoking rectangle
        ann_info = anns[idx]
        sx,sy,w,h = ann_info['bbox']
        
        image_det = image[sy:sy+h, sx:sx+w, :]
        
        # try:
        #     image_det = cv2.cvtColor(image_det, cv2.COLOR_BGR2GRAY)
        # except:
        #     continue
        if image_det is None:
            continue
        try:
            out_img_path = os.path.join(OUT_ROOT_DIR, os.path.basename(img_path))
            cv2.imwrite(out_img_path, image_det)
        except:
            print(sy, sy+h, sx, sx+w)
            continue
        
        if idx%100 == 0:
            print("{}/{}".format(idx, len(idx_list)))

    #     # show smoking points
    #     points = np.array(ann_info['keypoints']).reshape(-1, 3)
    #     for p in points:
    #         x,y,_ = p
    #         if x==-1 or y==-1:
    #             continue
    #         cv2.circle(image, (x,y), 2, (0,0,255), 2)
        
    #     cv2.rectangle(image, (sx,sy), (sx+w,sy+h), (95,214,38), 1)
    #     cv2.imshow('1', image)
    #     if cv2.waitKey(0) == ord('q'):
    #         break
    
    # cv2.destroyAllWindows()

