import os
import glob
import shutil

VIDEO_ROOT_DIR = r'/home/ruiming/workspace/pro/smoke_keypoints/smoke_keypoint/data/collect'

videos = os.listdir(VIDEO_ROOT_DIR)
duplist = []

for v in videos:
    if '.avi' not in v:
        continue
    basename = os.path.splitext(v)[0]
    checkname = basename+'.264.avi'

    if os.path.exists(os.path.join(VIDEO_ROOT_DIR, checkname)):
        #duplist.append(checkname)
        os.remove(os.path.join(VIDEO_ROOT_DIR, checkname))

#print(duplist)
#print(len(duplist))