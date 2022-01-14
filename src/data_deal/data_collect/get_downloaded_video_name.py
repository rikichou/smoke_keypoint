import os
import json
import glob
from tkinter.font import names

out_json_file_path = '/home/ruiming/workspace/pro/smoke_keypoints/smoke_keypoint/data/collect_chepai_names.json'
VIDEO_ROOT_DIR = '/home/ruiming/workspace/pro/smoke_keypoints/smoke_keypoint/data/collect_chepai'
file_paths = glob.glob(os.path.join(VIDEO_ROOT_DIR, '*/*'))

file_paths = [os.path.basename(f) for f in file_paths]

save_info = {}
save_info['vnames'] = file_paths

with open(out_json_file_path, 'w') as fp:
    json.dump(save_info, fp)

