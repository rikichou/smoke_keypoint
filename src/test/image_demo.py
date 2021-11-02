# Copyright (c) OpenMMLab. All rights reserved.
import sys
import os
sys.path.append(os.path.join(os.getcwd(), '../common_utils'))
import os

import cv2

if sys.platform == "darwin":
    print('=>>>>load data from mac platform')
    yolov5_src = "/Users/zhourui/workspace/pro/source/yolov5"
elif sys.platform == 'win32':
    print('=>>>>load data from window platform')
    yolov5_src = r"D:\workspace\pro\source\yolov5"
else:
    print('=>>>>load data from linux platform')
    yolov5_src = "/home/ruiming/workspace/pro/source/yolov5"

sys.path.insert(0, yolov5_src)

from smoke_keypoint_python import smoke_keypoint

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

def test_imgs(img_dir, args):
    args.use_scrfd = False
    args.cpu = True

    # init face detection model
    if args.use_scrfd:
        from scrdf_python import scrfd
        fd = scrfd.ScrdfFaceDet(0.45,
                                model_path='../common_utils/scrdf_python/models/model.pth',
                                device='cpu' if args.cpu else 'cuda',
                                config='../common_utils/scrdf_python/models/scrfd_500m.py')
    else:
        from FaceDetection import FaceDetect
        args.weights = os.path.join(yolov5_src, 'weights/200_last.pt')
        args.imgsz = 640
        fd = FaceDetect(args=args)

    # init smoke keypoint model
    sk = smoke_keypoint.SmokeKeypoint('../common_utils/smoke_keypoint_python/models/litehrnet_18_smoke_keypoint_256x256/litehrnet_18_smoke_keypoint_256x256.py',
                                 '../common_utils/smoke_keypoint_python/models/litehrnet_18_smoke_keypoint_256x256/epoch_140.pth',
                                 device='cpu' if args.cpu else 'cuda')

    imgs = os.listdir(img_dir)
    for path in imgs:
        # check if is image
        if not is_image(path):
            continue

        # read image
        path = os.path.join(img_dir, path)
        frame = cv2.imread(path)
        if frame is None:
            print("Failed to read image ", path)
            continue

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
            bbox = fd.detect(image)
            # print(bbox)
            if len(bbox) != 4 or sum(bbox) < 1:
                print("cat not detect face")
                continue
            else:
                sx, sy, ex, ey = bbox

        # facial expression
        points = sk(image, (sx,sy,ex,ey))
        color_map = {0:(0,0,255), 1:(0,255,0), 2:(255,0,0)}
        points_str = ''
        for idx,p in enumerate(points):
            x, y, score = p
            points_str += '{:.2f}  '.format(score)
            if score < 0.3:
                continue

            cv2.circle(image, (int(x), int(y)), 1, color_map[idx], 2)

        # debug
        cv2.rectangle(image, (sx, sy), (ex, ey), (255, 0, 0), 10)
        cv2.putText(image, points_str, (sx, sy - 20),
                    0, 2, (0, 0, 255), 2)
        cv2.imshow('debug', image)
        if cv2.waitKey(0) & 0xff == ord('q'):
            break
    # debug
    cv2.destroyAllWindows()

IMG_DIR = '/Users/zhourui/workspace/pro/smoke_keypoint/src/test/val_images'

test_imgs(IMG_DIR, args)