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

import glob
import argparse
from multiprocessing import Process, Lock, Value
import cv2
import numpy as np
from pathlib import Path

from FaceDetection import FaceDetect

def parse_args():
    parser = argparse.ArgumentParser(description='Build file list')
    parser.add_argument(
        'src_folder', type=str, help='root directory for the frames')
    parser.add_argument(
        '--out-name', type=str, default='facerect.npy', help='save name')
    parser.add_argument(
        '--level',
        type=int,
        default=3,
        choices=[1, 2, 3],
        help='directory level of data')
    parser.add_argument(
        '--num_worker',
        type=int,
        default=10,
        help='num workers to preprocess')
    parser.add_argument('--weights',
                        default=r'D:\workspace\pro\source\yolov5\weights\200_last.pt',
                        help='experiment', type=str)
    parser.add_argument('--imgsz',
                        default=640,
                        help='experiment', type=int)
    parser.add_argument(
        '--cpu',
        action='store_true',
        default=False,
        help='whether to use cpu')

    args = parser.parse_args()
    return args

def process_paths(paths, args, lock, counter, total_length):
    # init face detection model
    faceDetect = FaceDetect(args=args)
    for path in paths:
        # get all images to process
        images_path = glob.glob(os.path.join(path, '*.jpg'))
        images_rect_dict = {}
        for idx,img_path in enumerate(images_path):
            # open and preprocess image
            #image = cv2.imread(img_path) # there is some problem in Chinese char
            image = cv2.imdecode(np.fromfile(img_path, dtype=np.uint8), 1)

            if image is None:
                print("!!!!!!!!!!!!!!!!!!!! Failed to read image ", img_path)
                assert (False)

            # get face rectangle, have no face rectangle, set None
            bbox = faceDetect.detect(image)
            # print(bbox)
            if len(bbox) != 4 or sum(bbox) < 1:
                images_rect_dict[os.path.basename(img_path)] = None
            else:
                sx, sy, ex, ey = bbox
                images_rect_dict[os.path.basename(img_path)] = [sx,sy,ex,ey]

                # debug
                # cv2.rectangle(image, (int(sx), int(sy)), (int(ex), int(ey)), (0,0,255), 1)
                # cv2.imshow('1', image)
                # cv2.waitKey(0)
            if idx%500 == 0:
                print("{}/{}".format(idx, len(images_path)))

        np.save(os.path.join(path, args.out_name), images_rect_dict)

        # counter
        lock.acquire()
        try:
            # p_bar.update(1)
            counter.value += 1
            if counter.value % 50 == 0:
                print(f"{counter.value}/{total_length} done.")
        finally:
            lock.release()


def multi_process(video_dirs, args):
    process_num = args.num_worker
    # check if is valid
    to_process_video_dirs = []
    for v in video_dirs:
        if not os.path.isdir(v):
            continue
        if os.path.exists(os.path.join(v, args.out_name)):
            continue
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
        pw = Process(target=process_paths,
                     args=(files[start_index:end_index], args, lock, counter, len(files)))
        pw.start()
        process_pool.append(pw)

    for p in process_pool:
        p.join()

def main():
    args = parse_args()

    # get all video folders
    video_dirs = glob.glob(os.path.join(args.src_folder, str(Path('*/'*args.level))))

    # check if dir is dealed
    to_deal_videos = []
    for idx,v in enumerate(video_dirs):
        if not os.path.exists(os.path.join(v, args.out_name)):
            to_deal_videos.append(v)
        if idx%1000 == 0:
            print("Check is dealed {}/{}".format(idx+1, len(video_dirs)))
    print("Found {} videos! {} videos not yet processed!".format(len(video_dirs), len(to_deal_videos)))

    # multi process
    multi_process(video_dirs, args)

if __name__ == '__main__':
    main()