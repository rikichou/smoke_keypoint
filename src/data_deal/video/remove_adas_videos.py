import os
import glob
import shutil
import tqdm
from multiprocessing import Pool
import cv2
import mmcv
import random

VIDEO_ROOT_DIR = r'/zhourui/workspace/pro/smoke_keypoint/data/video/selected'
OUT_ROOT_DIR = r'/zhourui/workspace/pro/smoke_keypoint/data/video/selected_adas'
if not os.path.exists(OUT_ROOT_DIR):
    os.makedirs(OUT_ROOT_DIR)

videos = os.listdir(VIDEO_ROOT_DIR)

random.shuffle(videos)

def data_deal(item):
    vname,idx = item
    if '.avi' not in vname:
        return
    vpath = os.path.join(VIDEO_ROOT_DIR, vname)
    cap = cv2.VideoCapture(vpath)
    while True:
        ret,frame = cap.read()
        if ret is None:
            break
        
        # size check
        h,w,c = frame.shape
        if h == 1080 and w == 1920 and c == 3:
            out_path = os.path.join(OUT_ROOT_DIR, vname)
            # shutil.copyfile(vpath, out_path)
            # os.remove(vpath)
            shutil.move(vpath, out_path)
            # cv2.imshow('1', frame)
            # if cv2.waitKey(0) == ord('q'):
            #     assert(False)
            # print(vname, frame.shape)
        break
        
num_worker = 10
if __name__ == '__main__':
    with Pool(num_worker) as pool:
        r = list(tqdm.tqdm(pool.imap(
            data_deal,
            zip(videos, range(len(videos))))))

