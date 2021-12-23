import os
import sys
sys.path.append('../../common_utils')
sys.path.append('../../common_utils/Facealign')
import numpy as np
import json
import glob
from pathlib import Path
import shutil
import cv2
from Facealign import FaceAlignment

level = 1

SRC_ROOT_IMG_DIR = r'H:\pro\smoke_keypoint\data\smoke_keypoint_mpii_relabel\images\negative'
SRC_COPY_ROOT_IMG_DIR = r'H:\pro\smoke_keypoint\data\smoke_keypoint_mpii\images\negative'

OUT_ROOT_ANNS_DIR = r'H:\pro\smoke_keypoint\data\smoke_keypoint_mpii_relabel\annotations'
if not os.path.exists(OUT_ROOT_ANNS_DIR):
    os.makedirs(OUT_ROOT_ANNS_DIR)
OUT_ANNS_PATH = os.path.join(OUT_ROOT_ANNS_DIR, 'negative.json')

fa = FaceAlignment.faceAlignment('../../common_utils/Facealign/models/multiScale_7_20210716.pkl')

# load facerect info
facerect_file_path = os.path.join(SRC_ROOT_IMG_DIR, 'facerect.npy')
if not os.path.exists(facerect_file_path):
    print("Error !", facerect_file_path)
rect_infos = np.load(facerect_file_path, allow_pickle=True).item()

def get_center_acc_facial_points(image, pts):
    h,w,_ = image.shape

    # get mouth points (13_left, 14_up, 15_right, 16_down)
    mouth = pts[13:17]
    sx = min(mouth[:, 0])
    sy = min(mouth[:, 1])
    ex = max(mouth[:, 0])
    ey = max(mouth[:, 1])

    # expand
    # 1. just expand 200 pixel
    res = 200
    cx = int(sx + (ex - sx) / 2)
    cy = int(sy + (ey - sy) / 2)
    sx = cx - res
    sy = cy - res
    ex = cx + res
    ey = cy + res

    # border check
    msx = int(max(0, sx))
    msy = int(max(0, sy))
    mex = int(min(w - 1, ex))
    mey = int(min(h - 1, ey))

    cx = int(msx + (mex - msx) / 2)
    cy = int(msy + (mey - msy) / 2)

    return cx,cy,msx,msy,mex,mey

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
    imgname = os.path.basename(img)
    out_json_info['image'] = 'negative/' + imgname

    # scale
    out_json_info['scale'] = 1.0

    # center(according to the face rect)
    if imgname not in rect_infos:
        print("Can not found image face rect ", imgname)
        continue
    facerect = rect_infos[imgname]

    image = cv2.imread(f)

    # get face points
    #st = time.time()
    eye, mouth = fa.forward_with_rect(image, facerect)
    #print("{} s".format(time.time()-st))
    pts_pre = np.concatenate((eye, mouth), axis=0)

    # get center points and rect
    cx,cy,detsx,detsy,detex,detey = get_center_acc_facial_points(image, pts_pre)

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

    # update
    out_json_info['center'] = [cx, cy]
    sx,sy,ex,ey = facerect
    out_json_info['facerect'] = [sx,sy,ex,ey]

    pts_list = list(pts_pre)
    pts_list_out = []
    for al in pts_list:
        ao = [int(item) for item in al]
        pts_list_out.append(ao)
    out_json_info['facial_pts'] = pts_list_out

    out_json_info['point_type'] = 'negative'

    out_mpii_json.append(out_json_info)

    if idx%1000 == 0:
        print("{}/{}".format(idx, len(files)))

with open(OUT_ANNS_PATH, 'w') as fp:
    json.dump(out_mpii_json, fp)

















