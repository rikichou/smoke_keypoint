import os
import glob

IMAGE_DIR = r'E:\software\haisi\RuyiStudio-2.0.46\workspace\smoke_keypoints\smoke_keypoints\data\images'
out_file_path = r'E:\software\haisi\RuyiStudio-2.0.46\workspace\smoke_keypoints\smoke_keypoints\data\images\image_list_all_with_prefix.txt'

images_list = glob.glob(os.path.join(IMAGE_DIR, '*.jpg'))

with open(out_file_path,'w') as fp:
    for imgpath in images_list:
        fp.write(imgpath+'\n')