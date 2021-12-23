import os
import cv2

img_root_dir = r'E:\software\haisi\RuyiStudio-2.0.46\workspace\smoke_keypoints\smoke_keypoints\data\images'
imgs = os.listdir(img_root_dir)
for idx,img in enumerate(imgs):
    if '.jpg' not in img:
        continue
    src_img = os.path.join(img_root_dir, img)
    image = cv2.imread(src_img, 0)
    image = cv2.resize(image, (128, 128))
    cv2.imwrite(src_img, image)
    
    print("{}/{}".format(idx, len(imgs)))