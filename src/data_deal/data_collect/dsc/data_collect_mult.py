import os
import shutil
import json

from multiprocessing import Process, Lock, Value

DATA_ROOT_DIR = r'W:\videosrc'
OUT_ROOT_DIR = r'E:\workspace\pro\smoke_keypoint\data\dsc\search_20220116'

def data_search(data_root_dir, idx):
    print("Start thread {}:{}".format(idx, data_root_dir))

    count = 0
    save_info = {}
    save_info['items']  =[]

    for root_dir,sub_dirs,files in os.walk(data_root_dir):
        for f in files:
            fpath = os.path.join(root_dir, f)
            if 'dsc' in fpath.lower() or 'adplus' in fpath.lower() or 'ad plus' in fpath.lower():
                if '烟' in fpath or 'smok' in fpath.lower():
                    if '264' == fpath[-3:] or 'avi' == fpath[-3:].lower() or 'mp4' == fpath[-3:].lower():
                        dst_path = os.path.join(OUT_ROOT_DIR, os.path.basename(fpath))
                        save_info['items'].append(fpath)
                        if not os.path.exists(dst_path):
                            print(fpath)
                            shutil.copy(fpath, dst_path)
                        count += 1
                        if count %100 == 0:
                            print(count)
    if count>=1:
        outpath = OUT_ROOT_DIR+'_'+os.path.basename(data_root_dir)+'.json'
        with open(outpath, 'w') as fp:
            json.dump(save_info, fp, ensure_ascii=False)

    print("End thread {} !!!! {}".format(idx, data_root_dir))

def main():
    # create out dir
    if not os.path.isdir(OUT_ROOT_DIR):
        os.makedirs(OUT_ROOT_DIR)

    # get all sub dirs
    files = os.listdir(DATA_ROOT_DIR)
    filters = ['dsm']
    dirs = [os.path.join(DATA_ROOT_DIR, f) for f in files if os.path.isdir(os.path.join(DATA_ROOT_DIR, f)) and f not in filters]

    print(dirs)

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

