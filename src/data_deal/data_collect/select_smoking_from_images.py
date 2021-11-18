import sys
import os
print(sys.platform)
if sys.platform == "darwin":
    print('=>>>>load data from mac platform')
    sys.path.append("/Users/zhourui/workspace/pro/source/yolov5")
elif sys.platform == 'win32':
    print('=>>>>load data from window platform')
    sys.path.append(r"D:\workspace\pro\source\yolov5")
else:
    print('=>>>>load data from linux platform')
    sys.path.append(r"/zhourui/workspace/pro/source/yolov5")

import sys
sys.path.append(os.path.join(os.getcwd(), '../../common_utils'))
import glob
import shutil
import argparse
import multiprocessing
from multiprocessing import Process, Lock, Value
import cv2
import numpy as np
from pathlib import Path

from smoke_keypoint_python import smoke_keypoint
from FaceDetection import FaceDetect
from Facealign import FaceAlignment

def parse_args():
    parser = argparse.ArgumentParser(description='Build file list')
    parser.add_argument(
        'src_folder', type=str, help='root directory for the frames')
    parser.add_argument(
        'out_folder', type=str, help='save name')
    parser.add_argument(
        '--level',
        type=int,
        default=1,
        choices=[1, 2, 3],
        help='directory level of data')
    parser.add_argument(
        '--num_worker',
        type=int,
        default=10,
        help='num workers to preprocess')
    parser.add_argument('--weights',
                        default=r'/zhourui/workspace/pro/source/yolov5/weights/200_last.pt',
                        help='experiment', type=str)
    parser.add_argument('--imgsz',
                        default=640,
                        help='experiment', type=int)
    parser.add_argument(
        '--cpu',
        action='store_true',
        default=False,
        help='whether to use cpu')
    parser.add_argument(
        '--ext',
        type=str,
        default='jpg',
        help='video file extensions')

    args = parser.parse_args()
    return args

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

def process_paths(paths, args, lock, counter, total_length):
    # init face detection model
    faceDetect = FaceDetect(args=args)
    model_name = 'restiny_coco_128x128'
    epoch = 210
    sk = smoke_keypoint.SmokeKeypoint('../common_utils/smoke_keypoint_python/models/{}/{}.py'.format(model_name, model_name),
                                 '../common_utils/smoke_keypoint_python/models/{}/epoch_{}.pth'.format(model_name, epoch),
                                 device='cpu' if args.cpu else 'cuda',
                                      grayscale=True)
    # face alignment
    fa = FaceAlignment.faceAlignment('../common_utils/Facealign/models/multiScale_7_20210716.pkl')

    for path in paths:
        img_path = path

        # check if run before
        _, imgname = img_path.rsplit('/', maxsplit=1)
        out_dir = args.out_folder
        out_img_path = os.path.join(out_dir, imgname)

        # open and preprocess image
        image = cv2.imdecode(np.fromfile(img_path, dtype=np.uint8), 1)
        if image is None:
            print("!!!!!!!!!!!!!!!!!!!! Failed to read image ", img_path)
            assert (False)

        # get face rectangle, have no face rectangle, set None
        bbox = faceDetect.detect(image)
        # print(bbox)
        if len(bbox) != 4 or sum(bbox) < 1:
            continue

        sx, sy, ex, ey = bbox

        # facical keypoints
        eye, mouth = fa.forward_with_rect(image, (sx, sy, ex, ey))
        pts_pre = np.concatenate((eye, mouth), axis=0)

        cx, cy, detsx, detsy, detex, detey = get_smoking_rect_by_facialpoints(image, pts_pre, (sx, sy, ex, ey))

        # get smoke keypoint
        points = sk(image, (detsx, detsy, detex, detey))

        # if one score greater than 0.6, or more than one greater than 0.3
        high_count = 0
        low_count = 0
        for p in points:
            x,y,score = p
            if score < 0.5:
                continue
            elif score >= 0.5 and score < 0.8:
                low_count += 1
            else:
                low_count += 1
                high_count += 1
        if low_count>=2 or high_count>=1:
            # save warning pictures
            shutil.copy(img_path, out_img_path)

        # counter
        lock.acquire()
        try:
            # p_bar.update(1)
            counter.value += 1
            if counter.value % 50 == 0:
                print(f"{counter.value}/{total_length} done.")
        finally:
            lock.release()


def multi_process(images, args):
    process_num = args.num_worker
    # start process
    files = images
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
        pw = Process(target=process_paths,
                     args=(files[start_index:end_index], args, lock, counter, len(files)))
        pw.start()
        process_pool.append(pw)

    for p in process_pool:
        p.join()

def main():
    args = parse_args()

    # get all video folders
    images = glob.glob(args.src_folder + str(Path('/*' * args.level)) + '.' +
              args.ext)

    # check if dir is dealed
    print("Found {} videos!".format(len(images)))

    if not os.path.exists(args.out_folder):
        os.makedirs(args.out_folder)

    # multi process
    multi_process(images, args)

multiprocessing.set_start_method('forkserver', force=True)
if __name__ == '__main__':
    main()