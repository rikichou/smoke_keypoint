import os
import shutil

from multiprocessing import Process, Lock, Value

DATA_ROOT_DIR = r'\\192.168.80.4\CV_storage6\workspace\sampleclips'
OUT_ROOT_DIR = r'E:\workspace\pro\smoke_keypoint\data\search'

def data_search(data_root_dir, idx):
    print("Start thread {}:{}".format(idx, data_root_dir))
    for root_dir,sub_dirs,files in os.walk(data_root_dir):
        for f in files:
            fpath = os.path.join(root_dir, f)

            if '264' == fpath[-3:] or 'avi' == fpath[-3:].lower() or 'mp4' == fpath[-3:].lower():

                if '吃东西' in root_dir \
                or 'eatting' in fpath.lower():
                    out_sub_dir = os.path.join(OUT_ROOT_DIR, 'eatting')
                    if not os.path.exists(out_sub_dir):
                        os.makedirs(out_sub_dir)
                    shutil.copy(fpath, os.path.join(out_sub_dir, f))
                    os.system('echo {} >> eatting_{}.txt'.format(fpath, idx))
                elif '喝水' in root_dir \
                or 'drink' in fpath.lower():
                    out_sub_dir = os.path.join(OUT_ROOT_DIR, 'drinking')
                    if not os.path.exists(out_sub_dir):
                        os.makedirs(out_sub_dir)
                    shutil.copy(fpath, os.path.join(out_sub_dir, f))
                    os.system('echo {} >> drinking_{}.txt'.format(fpath, idx))
    print("End thread {} !!!! {}".format(idx, data_root_dir))

def main():
    # create out dir
    if not os.path.isdir(OUT_ROOT_DIR):
        os.makedirs(OUT_ROOT_DIR)

    # get all sub dirs
    files = os.listdir(DATA_ROOT_DIR)
    dirs = [os.path.join(DATA_ROOT_DIR, f) for f in files if os.path.isdir(os.path.join(DATA_ROOT_DIR, f))]

    # multi process
    process_pool = []
    for idx,subdir in enumerate(dirs):
        pw = Process(target=data_search,
                        args=(subdir, idx))
        pw.start()
        process_pool.append(pw)

    for p in process_pool:
        p.join()                        

if __name__ == '__main__':
    main()

