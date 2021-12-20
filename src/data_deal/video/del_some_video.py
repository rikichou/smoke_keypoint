import os

SRC_ROOT_DIR = r'H:\pro\smoke_keypoint\data\video\selected'
DEL_ROOT_DIR = r'H:\pro\smoke_keypoint\data\video\selected_adas'

vs = os.listdir(DEL_ROOT_DIR)

for v in vs:
    delp = os.path.join(SRC_ROOT_DIR, v)
    if os.path.exists(delp):
        os.remove(delp)