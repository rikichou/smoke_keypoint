import os
import random
import json
import cv2
import numpy as np

IMG_ROOT_DIR = '/Users/zhourui/Downloads/smoking_test_some_images_35/images'
ANNS_ROOT_DIR = '/Users/zhourui/Downloads/smoking_test_some_images_35/json_v0'

anns = os.listdir(ANNS_ROOT_DIR)
random.shuffle(anns)

for ann in anns:
    if 'meta.json' == ann or '.json' not in ann:
        continue
    with open(os.path.join(ANNS_ROOT_DIR, ann), 'r') as fp:
        infos = json.load(fp)
    
    # read image
    img_name = infos['img_name']
    img_path = os.path.join(IMG_ROOT_DIR, img_name)
    image = cv2.imread(img_path)

    # read all anns
    for obj_dict in infos['shapes']:
        if obj_dict['type'] == 'points':
            points = np.array(obj_dict['points']).reshape(-1, 2)
            for x,y in points:
                cv2.circle(image, (int(x),int(y)), 1, (255,200,22), 2)
        elif obj_dict['type'] == 'rectangle':
            sx,sy,ex,ey = obj_dict['points']
            cv2.rectangle(image, (sx,sy), (ex, ey), (0,255,0), 1)

    cv2.imshow('1', image)

    if cv2.waitKey(0) == ord('q'):
        break

cv2.destroyAllWindows()