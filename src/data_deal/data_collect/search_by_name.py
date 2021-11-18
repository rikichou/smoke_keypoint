# -*- coding: utf-8 -*-
import os
import shutil

IMG_ROOT_DIR = r'\\192.168.80.4\CV_storage6\F_public\public\workspace\exchange\dsm\sampleclips'
OUT_ROOT_DIR = r'H:\pro\smoke_keypoint\data\test\collect\videos\collect'
if not os.path.exists(OUT_ROOT_DIR):
  os.makedirs(OUT_ROOT_DIR)
save_info = {}
save_info['items']  =[]

count = 0
for root_dir,sub_dirs,files in os.walk(IMG_ROOT_DIR):
    for f in files:
        fpath = os.path.join(root_dir, f)
        if '抽烟' in root_dir or 'smok' in fpath.lower():
          if '264' == fpath[-3:] or 'avi' == fpath[-3:] or 'mp4' == fpath:
            dst_path = os.path.join(OUT_ROOT_DIR, os.path.basename(fpath))
            if not os.path.exists(dst_path) and os.path.exists(fpath):
                shutil.copy(fpath, dst_path)
            count += 1
            if count %1000 == 0:
                print(count)
# with open(OUT_COCO_JSON_PATH, 'w') as fp:
#   json.dump(save_info, fp)
