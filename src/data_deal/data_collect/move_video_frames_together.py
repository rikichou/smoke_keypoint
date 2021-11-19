import os
import glob
import shutil
import tqdm
from multiprocessing import Pool

num_worker = 10
IMG_ROOT_DIR = r'/zhourui/workspace/pro/smoke_keypoint/data/collected/rawframes'
IMG_OUT_DIR = r'/zhourui/workspace/pro/smoke_keypoint/data/collected/negative'
if not os.path.exists(IMG_OUT_DIR):
    os.makedirs(IMG_OUT_DIR)

imgs = glob.glob(os.path.join(IMG_ROOT_DIR, '*/*.jpg'))
print("Total {} images".format(len(imgs)))

def deal(item):
    img, idx = item

    _, dname, imgname = img.rsplit('/', maxsplit=2)
    out_name = dname+'_'+imgname
    out_path = os.path.join(IMG_OUT_DIR, out_name)

    if not os.path.exists(out_path):
        shutil.copy(img, out_path)


with Pool(num_worker) as pool:
    r = list(tqdm.tqdm(pool.imap(
        deal,
        zip(imgs, range(len(imgs))))))


