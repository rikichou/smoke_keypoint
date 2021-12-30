# Copyright (c) OpenMMLab. All rights reserved.
import glob
import random
import sys
import os
sys.path.append(os.path.join(os.getcwd(), '../common_utils'))
sys.path.append('../common_utils/Facealign')
import os
import time

import cv2

if sys.platform == "darwin":
    print('=>>>>load data from mac platform')
    sys.path.append("/Users/zhourui/workspace/pro/source/yolov5")
    yolov5_src = "/Users/zhourui/workspace/pro/source/yolov5"
elif sys.platform == 'win32':
    print('=>>>>load data from window platform')
    sys.path.append(r"D:\workspace\pro\source\yolov5")
    yolov5_src = r"D:\workspace\pro\source\yolov5"
else:
    print('=>>>>load data from linux platform')
    sys.path.append("/home/ruiming/workspace/pro/source/yolov5")
    yolov5_src = "/home/ruiming/workspace/pro/source/yolov5"

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
args = TmpClass()

def is_image(name):
    img_ext = ['.jpg', '.jpeg', '.png', '.bmp', '.tif']
    prefix,ext = os.path.splitext(name)
    ext = ext.lower()
    if ext in img_ext:
        return True
    else:
        return False

def test_video(video_path, args):
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

    model_name = 'hourglasstiny_128x128'
    model_name = 'hourglassmiddle_coco_128x128'
    model_name = 'hourglasstiny_coco_128x128'
    #model_name = 'restiny_coco_128x128_old'
    model_name = 'restiny_coco_128x128'
    
    #model_name = 'hourglassbigger_128x128'
    #model_name = 'hourglassmiddle_128x128'
    #model_name = 'restiny_128x128'
    epoch = 210
    #epoch = 210
    sk = smoke_keypoint.SmokeKeypoint('../common_utils/smoke_keypoint_python/models/{}/{}.py'.format(model_name, model_name),
                                 '../common_utils/smoke_keypoint_python/models/{}/epoch_{}.pth'.format(model_name, epoch),
                                 device='cpu' if args.cpu else 'cuda',
                                      grayscale=True)

    # face alignment
    fa = FaceAlignment.faceAlignment('../common_utils/Facealign/models/multiScale_7_20210716.pkl')

    if True:
        cap = cv2.VideoCapture(video_path)
    else:
        cap = cv2.VideoCapture(0)
    while True:
        # read image
        ret, frame = cap.read()

        if frame is None:
            print("Failed to read image ")
            break

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
            # print(bbox)
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
            cv2.rectangle(image, (int(alignx), int(aligny)), (int(alignex), int(alighey)), (0, 0, 255), 5)
            for i in range(pts_pre.shape[0]):
                if i not in range(13, 17):
                    continue
                cv2.circle(image, (int(pts_pre[i][0]), int(pts_pre[i][1])), 3, (255, 0, 0), -1)
                cv2.putText(image, str(i), (int(pts_pre[i][0]), int(pts_pre[i][1])), font, 0.4, (255, 255, 255), 1)

            # gen det center points
            #cx, cy, detsx, detsy, detex, detey = get_center_acc_facial_points(image, pts_pre)
            cx, cy, detsx, detsy, detex, detey = get_smoking_rect_by_facialpoints(image, pts_pre, (sx, sy, ex, ey))

            cv2.circle(image, (int(cx), int(cy)), 2, (164, 183, 214), 2)
            cv2.rectangle(image, (detsx, detsy), (detex, detey), (0, 255, 0), 10)

            points = sk(image, (detsx, detsy, detex, detey))
            color_map = {0: (0, 0, 255), 1: (0, 255, 0), 2: (255, 0, 0)}
            points_str = ''
            smoke_count = 0
            for idx, p in enumerate(points):
                x, y, score = p
                points_str += '{:.2f}  '.format(score)
                if score < 0.7:
                    continue
                smoke_count += 1

                cv2.circle(image, (int(x), int(y)), 1, color_map[idx], 2)

            if smoke_count==3:
                print(points_str)
            cv2.rectangle(image, (sx, sy), (ex, ey), (255, 0, 0), 10)
            sk_sx, sk_sy, sk_ex, sk_ey = sk.det_area
            cv2.rectangle(image, (sk_sx, sk_sy), (sk_ex, sk_ey), (255, 255, 0), 10)
            cv2.putText(image, points_str, (sx, sy - 20),
                        0, 2, (0, 0, 255), 2)
            image = cv2.resize(image, (int(image.shape[1]/2), int(image.shape[0]/2)))
            cv2.imshow('debug', image)
            if smoke_count == 3:
                if cv2.waitKey(0) & 0xff == ord('q'):
                    break
            else:
                if cv2.waitKey(1) & 0xff == ord('q'):
                    break
    # debug
    cv2.destroyAllWindows()

video_path = r'E:\workspace\pro\smoke_keypoint\data\test\20211109taiyuanruimingshebei\error\戴口罩\fileId=172277.avi'
video_dir = r'H:\pro\smoke_keypoint\data\test\dsc\1118\avi'
video_dir = r'H:\pro\smoke_keypoint\data\test\hard_example\嘴巴咬住耳机线'
video_dir = r'E:\360Downloads'
video_list = glob.glob(video_dir+'\*.avi')
video_list.extend(glob.glob(video_dir+'\*.mp4'))

#random.shuffle(video_list)
for v in video_list:
    test_video(v, args)