# Copyright (c) OpenMMLab. All rights reserved.
import glob
import random
import sys
import os
sys.path.append(os.path.join(os.getcwd(), '../common_utils'))
sys.path.append('../common_utils/Facealign')
import os
import time
import json
from multiprocessing import Process, Lock, Value
import cv2
from PIL import Image

if sys.platform == "darwin":
    print('=>>>>load data from mac platform')
    yolov5_src = "/Users/zhourui/workspace/pro/source/yolov5"
elif sys.platform == 'win32':
    print('=>>>>load data from window platform')
    #yolov5_src = r"E:\workspace\pro\source\yolov5"
    yolov5_src = r"D:\workspace\pro\source\yolov5"
else:
    print('=>>>>load data from linux platform')
    yolov5_src = "/home/ruiming/workspace/pro/source/yolov5"

sys.path.append(yolov5_src)

import numpy as np
from smoke_keypoint_python import smoke_keypoint
from Facealign import FaceAlignment

#video_path = r'E:\workspace\pro\facialExpression\data\test\NIR\20210810\0000000000000000-210810-201218-201303-000006000190.avi'
def get_smoking_rect_by_facialpoints(image, pts, facerect):
    shape = image.shape
    if len(shape) == 3:
        h,w,_ = shape
    else:
        h,w = shape
    # get mouth points (13_left, 14_up, 15_right, 16_down)
    mouth = pts[13:17]
    mouth = np.array(mouth)
    sx = min(mouth[:, 0])
    sy = min(mouth[:, 1])
    ex = max(mouth[:, 0])
    ey = max(mouth[:, 1])

    # border check
    msx = int(max(0, sx))
    msy = int(max(0, sy))
    mex = int(ex)
    mey = int(ey)

    cx = int(msx + (mex - msx) / 2)
    cy = int(msy + (mey - msy) / 2)

    # face rect
    ratio = 0.45
    face_sx, face_sy, face_ex, face_ey = facerect
    face_h = face_ey-face_sy
    face_w = face_ex-face_sx
    face_size = max(face_w, face_h)
    det_size = int(face_size*ratio)

    det_sx = max(0, cx - det_size)
    det_ex = min(w, cx + det_size)
    det_sy = max(0, cy - det_size)
    det_ey = min(h, cy + det_size)

    return cx, cy, det_sx, det_sy, det_ex, det_ey

def get_center_acc_facial_points(image, pts, res=100):
    h,w,_ = image.shape

    # get mouth points (13_left, 14_up, 15_right, 16_down)
    mouth = pts[13:17]
    sx = min(mouth[:, 0])
    sy = min(mouth[:, 1])
    ex = max(mouth[:, 0])
    ey = max(mouth[:, 1])

    # expand
    # 1. just expand 200 pixel
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

#
class TmpClass(object):
    def __init__(self):
        super().__init__()


def is_image(name):
    img_ext = ['.jpg', '.jpeg', '.png', '.bmp', '.tif']
    prefix,ext = os.path.splitext(name)
    ext = ext.lower()
    if ext in img_ext:
        return True
    else:
        return False

def test_video(videos, args):
    threaholds = 0.4
    args.use_scrfd = False
    args.cpu = False

    # init face detection model
    if args.use_scrfd:
        from scrdf_python import scrfd
        fd = scrfd.ScrdfFaceDet(0.45,
                                model_path='../utils/scrdf_python/models/model.pth',
                                device='cpu' if args.cpu else 'cuda',
                                config='../utils/scrdf_python/models/scrfd_500m.py')
    else:
        from FaceDetection import FaceDetect
        args.weights = os.path.join(yolov5_src, 'weights/200_last.pt')
        args.imgsz = 640
        fd = FaceDetect(args=args)

    # init smoke keypoint model
    model_name = 'restiny_128x128x3'
    epoch = 92
    sk = smoke_keypoint.SmokeKeypoint('../common_utils/smoke_keypoint_python/models/{}/{}.py'.format(model_name, model_name),
                                 '../common_utils/smoke_keypoint_python/models/{}/epoch_{}.pth'.format(model_name, epoch),
                                 device='cpu' if args.cpu else 'cuda',
                                      grayscale=False)
    # face alignment
    fa = FaceAlignment.faceAlignment('../common_utils/Facealign/models/multiScale_7_20210716.pkl')

    # video deal
    for video_path in videos:
        cap = cv2.VideoCapture(video_path)
        vname = os.path.basename(video_path)
        nameprefix,_ = os.path.splitext(vname)
        outdir = os.path.join(args.out_root_dir, nameprefix)
        if not os.path.exists(outdir):
            os.makedirs(outdir)
        scores_video = []
        i = 0
        while True:
            # read image
            ret, frame = cap.read()
            if frame is None:
                print("Failed to read image ")
                break
            i += 1
            # face detection
            image = frame
            if args.use_scrfd:
                result = fd.forward(image)
                if len(result) < 1:
                    continue
                box = result[0]
                sx, sy, ex, ey, prob = box
                if prob < 0.45:
                    continue
            else:
                st = time.time()
                bbox = fd.detect(image)
                if len(bbox) != 4 or sum(bbox) < 1:
                    print("cat not detect face")
                    continue
                else:
                    sx, sy, ex, ey = bbox

                # face align test
                eye, mouth = fa.forward_with_rect(image, (sx, sy, ex, ey))
                pts_pre = np.concatenate((eye, mouth), axis=0)

                font = cv2.FONT_HERSHEY_SIMPLEX
                alignx, aligny, alignex, alighey = fa.det_area
                cx, cy, detsx, detsy, detex, detey = get_smoking_rect_by_facialpoints(image, pts_pre, (sx, sy, ex, ey))

                # get smoke-keypoints
                points = sk(image, (detsx, detsy, detex, detey))
                scores_frame = np.array(list(np.array(points)[:,2]))
                if sum(scores_frame>=threaholds)==3:
                    # print(scores_frame)
                    frame = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                    im = Image.fromarray(np.uint8(frame))
                    im.save(os.path.join(outdir, 'img_{:05d}_{:.2f}_{:.2f}_{:.2f}.jpg'.format(i+1, scores_frame[0], scores_frame[1], scores_frame[2])))
    return scores_video

def multi_process(videos, args):
    process_num = args.num_worker
    # check if is valid
    to_process_video_dirs = []
    for v in videos:
        to_process_video_dirs.append(v)

    # start process
    files = to_process_video_dirs
    grid_size = len(files) // process_num
    process_pool = []
    lock = Lock()
    counter = Value("i", 0)
    for i in range(process_num):
        start_index = grid_size * i
        if i != process_num - 1:
            end_index = grid_size * (i + 1)
        else:
            end_index = len(files)
        pw = Process(target=test_video,
                     args=(files[start_index:end_index], args))
        pw.start()
        process_pool.append(pw)

    for p in process_pool:
        p.join()

if __name__ == '__main__':
    # get videos
    video_dir = r'E:\workspace\pro\smoke_keypoint\data\test\search_20220104_negative'
    video_list = glob.glob(video_dir+'\*.avi')
    video_list.extend(glob.glob(video_dir+'\*.mp4'))

    # args
    args = TmpClass()
    args.out_root_dir = r'E:\workspace\pro\smoke_keypoint\data\test\images\negative'
    if not os.path.exists(args.out_root_dir):
        os.makedirs(args.out_root_dir)
    args.num_worker = 6
    multi_process(video_list, args)

