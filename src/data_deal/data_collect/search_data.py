# -*- coding: utf-8 -*-
import os
import json

IMG_ROOT_DIR = r'\\192.168.80.4\CV_storage6\F_public\public\workspace\exchange\dsm\sampleclips'
OUT_COCO_JSON_PATH = r'H:\pro\smoke_keypoint\data\test\collect\videos\smoking.json'

def smoking_check_aitxt(aitxt_path):
  with open(aitxt_path, "r") as fp:
    frame_idx = -1
    for txt_line in fp.readlines():
      txt_line = txt_line.strip()
      if "frameindexinmp4:" in txt_line:
        frame_idx = int(txt_line.split(":")[1])
      elif "grp9:100,200:FATIGUE" in txt_line or "grp1:100,200:FATIGUE" in txt_line:
        if frame_idx == -1:
          continue
        ret_list.append(frame_idx)  # 记录疲劳报警是视频中的第几帧
def get_fatigue_index_from_aitxt(filepath):
  """
  解析Aitext记录的信息
  :param filepath: (str) file path
  :return
      ret_list: (list) fatigue warning frame index
  """
  ret_list = []
  with open(filepath, "r") as fp:
    frame_idx = -1
    for txt_line in fp.readlines():
      txt_line = txt_line.strip()
      if "frameindexinmp4:" in txt_line:
        frame_idx = int(txt_line.split(":")[1])
      elif "grp9:100,200:FATIGUE" in txt_line or "grp1:100,200:FATIGUE" in txt_line:
        if frame_idx == -1:
          continue
        ret_list.append(frame_idx)  # 记录疲劳报警是视频中的第几帧

  return ret_list

save_info = {}
save_info['items']  =[]

for root_dir,sub_dirs,files in os.walk(IMG_ROOT_DIR):
    for f in files:
        # check if aitxt file
        if f[-5:].lower() == 'aitxt':
          fpath = os.path.join(root_dir, f)

          # check if h264 or avi
          video_path = None
          path_264 = fpath[:-5] + 'h264'
          path_avi = fpath[:-5] + 'avi'
          path_mp4 = fpath[:-5] + 'mp4'
          if os.path.exists(path_264):
              video_path = path_264
          elif os.path.exists(path_avi):
              video_path = path_avi
          elif os.path.exists(path_mp4):
              video_path = path_mp4
          else:
              continue

          # parse aitxt info
          if smoking_check_aitxt(aitxt_path=fpath):
            save_info['items'].append([fpath, video_path])

with open(OUT_COCO_JSON_PATH, 'w') as fp:
  json.dump(save_info, fp)
