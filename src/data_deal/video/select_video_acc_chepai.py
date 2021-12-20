import os
import glob
import random
import shutil
import tqdm
from multiprocessing import Pool

MAX_NUM_PER_LICENSE_PLATE = 20

VIDEO_ROOT_DIR = r'/home/ruiming/workspace/pro/smoke_keypoints/smoke_keypoint/data/collect_chepai'
OUT_VIDEO_ROOT_DIR = r'/home/ruiming/workspace/pro/smoke_keypoints/smoke_keypoint/data/selected'
if not os.path.exists(OUT_VIDEO_ROOT_DIR):
    os.makedirs(OUT_VIDEO_ROOT_DIR)

dirs = os.listdir(VIDEO_ROOT_DIR)

total = 0

def data_deal(item):
    d,_ = item
    # read all video list
    dpath = os.path.join(VIDEO_ROOT_DIR, d)
    if not os.path.isdir(dpath):
        return
    videos = os.listdir(dpath)
    random.shuffle(videos)
    videos = videos[:MAX_NUM_PER_LICENSE_PLATE]

    # copy to dst path
    for v in videos:
        src_path = os.path.join(dpath, v)
        dst_path = os.path.join(OUT_VIDEO_ROOT_DIR, v)
        shutil.copy(src_path, dst_path)


num_worker = 20
if __name__ == '__main__':
    with Pool(num_worker) as pool:
        r = list(tqdm.tqdm(pool.imap(
            data_deal,
            zip(dirs, range(len(dirs))))))





    
