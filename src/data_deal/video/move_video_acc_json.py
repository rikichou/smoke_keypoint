import os
import json
import shutil
import tqdm
from multiprocessing import Pool

JSON_PATH = '/home/ruiming/workspace/pro/smoke_keypoints/smoke_keypoint/data/collect.json'
SRC_ROOT_DIR = '/home/ruiming/workspace/pro/smoke_keypoints/smoke_keypoint/data/collect'
OUT_ROOT_DIR = '/home/ruiming/workspace/pro/smoke_keypoints/smoke_keypoint/data/collect_dealed_name'
if not os.path.exists(OUT_ROOT_DIR):
    os.makedirs(OUT_ROOT_DIR)

with open(JSON_PATH, 'r') as fp:
    infos = json.load(fp)

def data_deal(item):
    key,idx = item
    for f in infos[key]:
        f = os.path.join(SRC_ROOT_DIR, os.path.basename(f))
        if not os.path.exists(f):
            continue
        # copy to dest
        dst_dir = os.path.join(OUT_ROOT_DIR, key)    
        if not os.path.exists(dst_dir):
            os.makedirs(dst_dir)
        dst_path = os.path.join(dst_dir, os.path.basename(f))
        #shutil.copy(f, dst_path)
        #os.remove(f)
        shutil.move(f, dst_path)


num_worker = 20
if __name__ == '__main__':

    with Pool(num_worker) as pool:
        r = list(tqdm.tqdm(pool.imap(
            data_deal,
            zip(infos, range(len(infos))))))

