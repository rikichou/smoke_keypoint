import os
import json

JSON_PATH = '/home/ruiming/workspace/pro/source/PaddleOCR/temp.json'
OUT_ROOT_DIR = '/home/ruiming/workspace/pro/smoke_keypoints/smoke_keypoint/data/collect'

with open(JSON_PATH, 'r') as fp:
    infos = json.load(fp)

print(infos)