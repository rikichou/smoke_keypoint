import os
import json
import shutil

JSONN_PATH = r'E:\workspace\pro\smoke_keypoint\data\name.json'
VIDEO_ROOT_DIR = r'E:\workspace\pro\smoke_keypoint\data\search'
VIDEO_OUT_DIR = r'E:\workspace\pro\smoke_keypoint\data\search_20220104'
if not os.path.exists(VIDEO_OUT_DIR):
    os.makedirs(VIDEO_OUT_DIR)

with open(JSONN_PATH, 'r') as fp:
    infos = json.load(fp)['names']

vs = os.listdir(VIDEO_ROOT_DIR)

new = []
dup = []
for v in vs:
    if '.h264' in v:
        newv = v[:v.find('.h264')+5]
    elif '.264' in v:
        newv = v[:v.find('.264')+4]
    else:
        newv = v
    # rsplit
    prefix,ext = newv.rsplit('.', maxsplit=1)
    
    if prefix not in infos:
        new.append(v)
        shutil.move(os.path.join(VIDEO_ROOT_DIR, v), os.path.join(VIDEO_OUT_DIR, newv))
    else:
        dup.append(v)
        
print("Total {}, new {}".format(len(vs), len(new)))

# for d in dup:
#     print(d)