import cv2

image = cv2.imread(r'E:\software\haisi\RuyiStudio-2.0.46\workspace\smoke_keypoints\smoke_keypoints\data\test_images\1.jpg')
image = cv2.resize(image, (128, 128))
cv2.imwrite(r'E:\software\haisi\RuyiStudio-2.0.46\workspace\smoke_keypoints\smoke_keypoints\data\test_images\test.jpg', image)