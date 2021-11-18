import argparse
import glob
import os
import os.path as osp
import sys
import warnings
from multiprocessing import Pool

import tqdm
import cv2
from PIL import Image
import mmcv

from pathlib import Path
import numpy as np

skip_count = 0
skip_dict = {}
def extract_frame(vid_item):
    """Generate optical flow using dense flow.

    Args:
        vid_item (list): Video item containing video full path,
            video (short) path, video id.

    Returns:
        bool: Whether generate optical flow successfully.
    """
    global args
    global skip_count
    full_path, vid_path, vid_id = vid_item

    if str(Path('/')) in vid_path:
        out_full_path = osp.join(args.out_dir, vid_path.rsplit(str(Path('/')), maxsplit=1)[0])
    else:
        out_full_path = args.out_dir

    # Not like using denseflow,
    # Use OpenCV will not make a sub directory with the video name
    video_name = osp.splitext(osp.basename(vid_path))[0]
    if '01p015000000' in video_name or '_CH1' in video_name or 'channelNo0.' in video_name:
        return True
    out_full_path = osp.join(out_full_path, video_name)

    # check if it is dealed
    vr = mmcv.VideoReader(full_path)
    if not os.path.exists(out_full_path):
        os.makedirs(out_full_path)
    else:
        total_count = len(glob.glob(os.path.join(out_full_path, "*.jpg")))
        frame_cnt = vr.frame_cnt
        if frame_cnt <= total_count:
            skip_count += 1
            tmp_name = vid_path.rsplit(str(Path('/')), maxsplit=1)[0]
            if not tmp_name in skip_dict:
                skip_dict[tmp_name] = 0
            skip_dict[tmp_name] += 1
            if skip_count%1000 == 0:
                print(skip_dict)
                sys.stdout.flush()
            return True

    # for i in range(len(vr)):
    for i, vr_frame in enumerate(vr):
        if vr_frame is not None:
            # w, h, _ = np.shape(vr_frame)
            # if args.new_short == 0:
            #     if args.new_width == 0 or args.new_height == 0:
            #         # Keep original shape
            #         out_img = vr_frame
            #     else:
            #         out_img = mmcv.imresize(vr_frame,
            #                                 (args.new_width,
            #                                  args.new_height))
            # else:
            #     if min(h, w) == h:
            #         new_h = args.new_short
            #         new_w = int((new_h / h) * w)
            #     else:
            #         new_w = args.new_short
            #         new_h = int((new_w / w) * h)
            #     out_img = mmcv.imresize(vr_frame, (new_h, new_w))
            # mmcv.imwrite(vr_frame,
            #              f'{out_full_path}/img_{i + 1:05d}.jpg')
            frame = cv2.cvtColor(vr_frame, cv2.COLOR_BGR2GRAY)
            im = Image.fromarray(np.uint8(frame))

            im.save(os.path.join(out_full_path, 'img_{:05d}.jpg'.format(i+1)))
        else:
            warnings.warn(
                'Length inconsistent!'
                f'Early stop with {i + 1} out of {len(vr)} frames.')
            break

    sys.stdout.flush()
    return True

def is_video(name):
    _,ext = os.path.splitext(name)
    ext = ext.lower()
    if ext in ['.avi', '.mp4', '.webm']:
        return True
    else:
        return False

def parse_args():
    parser = argparse.ArgumentParser(description='extract optical flows')
    parser.add_argument('src_dir', type=str, help='source video directory')
    parser.add_argument('out_dir', type=str, help='output rawframe directory')
    parser.add_argument(
        '--level',
        type=int,
        choices=[1, 2, 3],
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
        default='avi',
        choices=['avi', 'mp4', 'webm'],
        help='video file extensions')
    parser.add_argument(
        '--mixed-ext',
        action='store_true',
        help='process video files with mixed extensions')
    parser.add_argument(
        '--new-width', type=int, default=0, help='resize image width')
    parser.add_argument(
        '--new-height', type=int, default=0, help='resize image height')
    parser.add_argument(
        '--new-short',
        type=int,
        default=0,
        help='resize image short side length keeping ratio')

    args = parser.parse_args()

    return args

args = parse_args()
if __name__ == '__main__':
    if not osp.isdir(args.out_dir):
        print(f'Creating folder: {args.out_dir}')
        os.makedirs(args.out_dir)

    print('Reading videos from folder: ', args.src_dir)
    if args.mixed_ext:
        print('Extension of videos is mixed')
        fullpath_list = glob.glob(args.src_dir + str(Path('/*' * args.level)))
        done_fullpath_list = glob.glob(args.out_dir + str(Path('/*' * args.level)))
        fullpath_list = [x for x in fullpath_list if is_video(x)]
    else:
        print('Extension of videos: ', args.ext)
        fullpath_list = glob.glob(args.src_dir + str(Path('/*' * args.level)) + '.' +
                                  args.ext)
        done_fullpath_list = glob.glob(args.out_dir + str(Path('/*' * args.level)))
    print('Total number of videos found: ', len(fullpath_list))

    if args.level == 3:
        vid_list = list(
            map(
                lambda p: os.path.join(*(p.rsplit(str(Path('/')), maxsplit=3)[-3:])),
                fullpath_list))
    if args.level == 2:
        vid_list = list(
            map(
                lambda p: os.path.join(*(p.rsplit(str(Path('/')), maxsplit=2)[-2:])),
                fullpath_list))
    elif args.level == 1:
        vid_list = list(map(osp.basename, fullpath_list))

    with Pool(args.num_worker) as pool:
        r = list(tqdm.tqdm(pool.imap(
            extract_frame,
            zip(fullpath_list, vid_list, range(len(vid_list))))))
