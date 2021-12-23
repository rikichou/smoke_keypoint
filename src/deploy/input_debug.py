# 对于新加入的样本：
#   对于标注过的正样本
#       1，首先通过get_facerect.py生成人脸框信息
#       2，然后通过convert_to_mpii_json_format.py来将标注信息转换为mpii格式的json
#       3，然后运行update_json_center来将人脸框信息融入到mpii json信息中
#   对于负样本
#       1，首先通过get_facerect.py生成人脸框信息
#       2，然后通过normal_data_gen_jsonfile来生成mpii格式的json信息
#   最后：
#       1，融合正样本和负样本的mpii json文件
#       2，划分训练集和验证集

import sys
import os
import random
import cv2
import json
import numpy as np

from PIL import Image

# ANNS_FILE_PATH = r'H:\pro\smoke_keypoint\data\smoke_keypoint_mpii_relabel\annotations\positive.json'
# IMG_ROOT_DIR = r'H:\pro\smoke_keypoint\data\smoke_keypoint_mpii_relabel\images'
#
# OUT_STA_PATH = r'H:\pro\smoke_keypoint\data\smoke_keypoint_mpii_relabel\annotations\statistics_positive.npy'
#
# with open(ANNS_FILE_PATH, 'r') as fp:
#     anns = json.load(fp)
#
# img = []
# valid_ann = []
# invalid_ann = []
# for ann in anns:
#     if ann['point_type'] == 'negative':
#         invalid_ann.append(ann)
#     else:
#         if ann['image'] not in img:
#             valid_ann.append(ann)
#             img.append(ann['image'])
#         else:
#             invalid_ann.append(ann)
#
# print("valid {}, invalid {}".format(len(valid_ann), len(invalid_ann)))
#
# with open(r'H:\pro\smoke_keypoint\data\smoke_keypoint_mpii_relabel\annotations\valid_positive.json', 'w') as fp:
#     json.dump(valid_ann, fp)
#
# with open(r'H:\pro\smoke_keypoint\data\smoke_keypoint_mpii_relabel\annotations\invalid_positive.json', 'w') as fp:
#     json.dump(invalid_ann, fp)

# import cv2

# img = cv2.imread(r'H:\pro\smoke_keypoint\data\test\tmp\1')
# print(img)


# import time

# path = r'H:\pro\smoke_keypoint\data\video\rawframes\_A0H97800000000-210505-094345-094355-01p016000000'
# file_ct_time = time.asctime(time.localtime(os.path.getctime(path)))
# print(file_ct_time)

# file_ct_time = time.localtime(os.path.getctime(path))
# print(file_ct_time.tm_hour, file_ct_time.tm_min, file_ct_time.tm_sec)


import os
import cv2
import numpy as np

def _get_3rd_point(a, b):
    """To calculate the affine matrix, three pairs of points are required. This
    function is used to get the 3rd point, given 2D points a & b.

    The 3rd point is defined by rotating vector `a - b` by 90 degrees
    anticlockwise, using b as the rotation center.

    Args:
        a (np.ndarray): point(x,y)
        b (np.ndarray): point(x,y)

    Returns:
        np.ndarray: The 3rd point.
    """
    assert len(a) == 2
    assert len(b) == 2
    direction = a - b
    third_pt = b + np.array([-direction[1], direction[0]], dtype=np.float32)

    return third_pt


def rotate_point(pt, angle_rad):
    """Rotate a point by an angle.

    Args:
        pt (list[float]): 2 dimensional point to be rotated
        angle_rad (float): rotation angle by radian

    Returns:
        list[float]: Rotated point.
    """
    assert len(pt) == 2
    sn, cs = np.sin(angle_rad), np.cos(angle_rad)
    new_x = pt[0] * cs - pt[1] * sn
    new_y = pt[0] * sn + pt[1] * cs
    rotated_pt = [new_x, new_y]

    return rotated_pt

def get_affine_transform(center,
                         scale,
                         rot,
                         output_size,
                         shift=(0., 0.),
                         inv=False):
    """Get the affine transform matrix, given the center/scale/rot/output_size.

    Args:
        center (np.ndarray[2, ]): Center of the bounding box (x, y).
        scale (np.ndarray[2, ]): Scale of the bounding box
            wrt [width, height].
        rot (float): Rotation angle (degree).
        output_size (np.ndarray[2, ] | list(2,)): Size of the
            destination heatmaps.
        shift (0-100%): Shift translation ratio wrt the width/height.
            Default (0., 0.).
        inv (bool): Option to inverse the affine transform direction.
            (inv=False: src->dst or inv=True: dst->src)

    Returns:
        np.ndarray: The transform matrix.
    """
    assert len(center) == 2
    assert len(scale) == 2
    assert len(output_size) == 2
    assert len(shift) == 2

    # pixel_std is 200.
    scale_tmp = scale * 200.0

    shift = np.array(shift)
    src_w = scale_tmp[0]
    dst_w = output_size[0]
    dst_h = output_size[1]

    rot_rad = np.pi * rot / 180
    src_dir = rotate_point([0., src_w * -0.5], rot_rad)
    dst_dir = np.array([0., dst_w * -0.5])

    src = np.zeros((3, 2), dtype=np.float32)
    src[0, :] = center + scale_tmp * shift
    src[1, :] = center + src_dir + scale_tmp * shift
    src[2, :] = _get_3rd_point(src[0, :], src[1, :])

    dst = np.zeros((3, 2), dtype=np.float32)
    dst[0, :] = [dst_w * 0.5, dst_h * 0.5]
    dst[1, :] = np.array([dst_w * 0.5, dst_h * 0.5]) + dst_dir
    dst[2, :] = _get_3rd_point(dst[0, :], dst[1, :])

    if inv:
        trans = cv2.getAffineTransform(np.float32(dst), np.float32(src))
    else:
        trans = cv2.getAffineTransform(np.float32(src), np.float32(dst))

    return trans

def _box2cs(box):
    """This encodes bbox(x,y,w,h) into (center, scale)

    Args:
        x, y, w, h

    Returns:
        tuple: A tuple containing center and scale.

        - np.ndarray[float32](2,): Center of the bbox (x, y).
        - np.ndarray[float32](2,): Scale of the bbox w & h.
    """

    x, y, w, h = box[:4]
    input_size = [128,128]
    aspect_ratio = input_size[0] / input_size[1]
    center = np.array([x + w * 0.5, y + h * 0.5], dtype=np.float32)

    if w > aspect_ratio * h:
        h = w * 1.0 / aspect_ratio
    elif w < aspect_ratio * h:
        w = h * aspect_ratio

    # pixel std is 200.0
    scale = np.array([w / 200.0, h / 200.0], dtype=np.float32)

    scale = scale * 1.25

    return center, scale

imgpath = r'E:\software\nvtai_tool\config\facial_expression\cnn25\data\test\00001.jpg'
image_org = cv2.imread(imgpath)

image = image_org[:100,:200,:]

c, s = _box2cs((0, 0, image.shape[1], image.shape[0]))
r = 0

s = np.array([1.0, 1.0])

image_size = np.array([128,128])
trans = get_affine_transform(c, s, r, image_size)
img = cv2.warpAffine(
    image,
    trans, (int(image_size[0]), int(image_size[1])),
    flags=cv2.INTER_LINEAR)

cv2.imshow('org', image_org)
cv2.imshow('image', image)
cv2.imshow('trans', img)

cv2.waitKey(0)
cv2.destroyAllWindows()
