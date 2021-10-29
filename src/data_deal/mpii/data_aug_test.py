import cv2
import numpy as np

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



import sys
import os

import cv2
import json

ANNS_FILE_PATH = '/Volumes/10.20.132.160-1/pro/smoke_keypoint/data/mpii/untar/annotations/mpii_val.json'
IMG_ROOT_DIR = '/Volumes/10.20.132.160-1/pro/smoke_keypoint/data/mpii/untar/images'

with open(ANNS_FILE_PATH, 'r') as fp:
    anns = json.load(fp)

rotation = 0
image_size = np.array([256, 256])

for ann in anns:
    # get image
    imgname = ann['image']
    img_path = os.path.join(IMG_ROOT_DIR, imgname)
    img = cv2.imread(img_path)

    # write points
    points = ann['joints']
    for p in points:
        x,y = p
        cv2.circle(img, (int(x),int(y)), 1, (0,0,255), 2)
    cx,cy = ann['center']
    center = np.array(ann['center'], dtype=np.float32)
    scale = np.array([ann['scale'], ann['scale']], dtype=np.float32)

    print("Scale ", scale)

    scale_tmp = scale[0] * 200.0

    print("Scale tmp ", scale_tmp)

    sx = int(cx -  scale_tmp/2)
    sy = int(cy - scale_tmp / 2)
    ex = int(cx + scale_tmp/2)
    ey = int(cy + scale_tmp / 2)

    cv2.circle(img, (int(cx), int(cy)), 1, (0, 255, 255), 5)
    img = cv2.rectangle(img, (sx,sy), (ex,ey), (0, 255, 255), 2)

    trans = get_affine_transform(center, scale, rotation, image_size)
    img_out = cv2.warpAffine(
        img,
        trans, (int(image_size[0]), int(image_size[1])),
        flags=cv2.INTER_LINEAR)
    print(img_out.shape)

    cv2.imshow('before', img)
    cv2.imshow('after', img_out)

    cv2.waitKey(0)