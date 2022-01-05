import argparse
import glob
import os
import os.path as osp
import sys
import warnings
from multiprocessing import Pool

import shutil
import tqdm
import time
import random
import cv2
from PIL import Image
import mmcv

from pathlib import Path
import numpy as np

def parse_args():
    parser = argparse.ArgumentParser(description='extract optical flows')
    parser.add_argument('src_dir', type=str, help='source video directory')
    parser.add_argument(
        '--level',
        type=int,
        choices=[1, 2, 3, 4],
        default=3,
        help='directory level of data')
    parser.add_argument(
        '--num-worker',
        type=int,
        default=8,
        help='number of workers to build rawframes')
    parser.add_argument(
        '--ext',
        type=str,
        default='264',
        choices=['264', 'avi', 'mp4', 'webm'],
        help='numpy file extensions')

    args = parser.parse_args()

    return args

def deal(item):
    # copy data
    f,idx = item
    name = os.path.basename(f)
    prefix,ext = os.path.splitext(f)

    new_name = ''
    for c in name:
        # check chinese code
        if u'\u4e00' <= c <= u'\u9fff':
            new_name += '_'
        elif c==' ':
            new_name += '_'
        elif ord(c) == 9560:
            new_name += '_'
        elif ord(c) == 9524:
            continue
        elif ord(c) == 9575:
            new_name += '_'
        elif ord(c) == 1094:
            continue
        else:
            new_name += c

    if new_name != name:

        if os.path.isdir(f):
            print("mv {} : {} ---> {}".format(os.path.dirname(f), name, new_name))
            #os.popen("mv \'{}\' {}".format(f, os.path.join(os.path.dirname(f), new_name))).readlines()
            #shutil.move(f, os.path.join(os.path.dirname(f), new_name))
        else:

            basename = time.strftime('%Y%m%d_%H%M%S')+str(random.randint(0,1000000))
            #new_video_name = os.path.join(os.path.dirname(f), new_name+basename)+ext
            new_video_name = os.path.join(os.path.dirname(f), new_name)

            print("rename: {} ---> {}".format(f, new_video_name))
            os.rename(f, new_video_name)
            # os.rename(aitxt_path, new_aitxt_name)

args = parse_args()
if __name__ == '__main__':
    print('Reading npy from folder: ', args.src_dir)

    fullpath_list = glob.glob(args.src_dir + str(Path('/*' * args.level))+'.'+args.ext)
    print('Total number of files: ', len(fullpath_list))


    with Pool(args.num_worker) as pool:
        r = list(tqdm.tqdm(pool.imap(
            deal,
            zip(fullpath_list, range(len(fullpath_list))))))
