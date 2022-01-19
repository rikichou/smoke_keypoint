import os
import sys
import glob
import cv2
from pathlib import Path
from multiprocessing import Process, Lock, Value
import shutil

utils_dir = r'E:\workspace\pro\fatigue\src\common_utils'
sys.path.append(utils_dir)
from facedet_scrfd_rgb import scrfd

num_worker = 4
VIDEO_ROOT_DIR = r'E:\workspace\pro\smoke_keypoint\data\dsc\search_20220115-20220116'
VIDEO_OUT_DIR = r'E:\workspace\pro\smoke_keypoint\data\dsc\search_20220115-20220116'
if not os.path.exists(VIDEO_OUT_DIR):
    os.makedirs(VIDEO_OUT_DIR)
VIDEO_OUT_DIR_ADAS = r'E:\workspace\pro\smoke_keypoint\data\dsc\search_20220115-20220116_adas'
if not os.path.exists(VIDEO_OUT_DIR_ADAS):
    os.makedirs(VIDEO_OUT_DIR_ADAS)

def process_paths(videos, lock, counter, total_length):
    # init face detection model
    fd = scrfd.ScrdfFaceDet(0.6, device='cuda', model_path='../../common_utils/facedet_scrfd_rgb/models/model.pth', config='../../common_utils/facedet_scrfd_rgb/models/scrfd_500m.py')

    for v in videos:
        count = 0
        face_count = 0
        cap = cv2.VideoCapture(v)
        while True:
            # read image
            ret, image = cap.read()
            if image is None:
                break
            # face det
            count += 1

            if count < 100:
                continue

            result = fd.forward(image)
            if len(result) >= 1:
                face_count += 1
                # box = result[0]
                # sx, sy, ex, ey, prob = box
            # check break
            if count >= 120:
                break
        # check if adas
        cap.release()
        
        if face_count<10:
            # move to dst dir
            if not os.path.exists(os.path.join(VIDEO_OUT_DIR_ADAS, os.path.basename(v))):
                shutil.move(v, os.path.join(VIDEO_OUT_DIR_ADAS, os.path.basename(v)))
        else:
            if not os.path.exists(os.path.join(VIDEO_OUT_DIR, os.path.basename(v))):
                shutil.move(v, os.path.join(VIDEO_OUT_DIR, os.path.basename(v)))
        # counter
        lock.acquire()
        try:
            # p_bar.update(1)
            counter.value += 1
            if counter.value % 100 == 0:
                print(f"{counter.value}/{total_length} done.")
        finally:
            lock.release()

def multi_process(videos):
    process_num = num_worker

    # start process
    files = videos
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
                     args=(files[start_index:end_index], lock, counter, len(files)))
        pw.start()
        process_pool.append(pw)

    for p in process_pool:
        p.join()

def main():
    # get all video folders
    videos_paths = glob.glob(os.path.join(VIDEO_ROOT_DIR, '*.avi'))

    # check if dir is dealed
    to_deal_videos = videos_paths
    print("Found {} videos!".format(len(to_deal_videos)))

    # multi process
    multi_process(to_deal_videos)

if __name__ == '__main__':
    main()