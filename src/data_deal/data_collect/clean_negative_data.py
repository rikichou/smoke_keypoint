import glob
import os
import shutil
import glob

ROOT_IMG_DIR = r'H:\pro\smoke_keypoint\data\smoke_keypoint_mpii_relabel\images\negative'
ERROR_IMG_DIR = r'H:\pro\smoke_keypoint\data\test\collect\error_ana\0'
OUT_IMG_ROOT_DIR = r'H:\pro\smoke_keypoint\data\smoke_keypoint_mpii_relabel\images\negative_clean'
if not os.path.exists(OUT_IMG_ROOT_DIR):
    os.makedirs(OUT_IMG_ROOT_DIR)

imgs = glob.glob(os.path.join(ERROR_IMG_DIR, '*.jpg'))

error_video_set = set()
for img in imgs:
    video_name = os.path.basename(img).rsplit('_', maxsplit=2)[0]
    error_video_set.add(video_name)

remove_list = []
imgs = glob.glob(os.path.join(ROOT_IMG_DIR, '*.jpg'))
for idx,img in enumerate(imgs):
    video_name = os.path.basename(img).rsplit('_', maxsplit=2)[0]
    if video_name in error_video_set:
        remove_list.append(img)
        shutil.copy(img, os.path.join(OUT_IMG_ROOT_DIR, os.path.basename(img)))
        os.remove(img)

    if idx%1000 == 0:
        print("{}/{}".format(idx, len(imgs)))
print("remove ", len(remove_list))