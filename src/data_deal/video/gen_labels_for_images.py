import sys
import os

sys.path.append(os.path.join(os.getcwd(), '../../common_utils'))
sys.path.append('../../common_utils/Facealign')

if sys.platform == "darwin":
    print('=>>>>load data from mac platform')
    yolov5_src = "/Users/zhourui/workspace/pro/source/yolov5"
elif sys.platform == 'win32':
    print('=>>>>load data from window platform')
    yolov5_src = r"D:\workspace\pro\source\yolov5"
else:
    print('=>>>>load data from linux platform')
    yolov5_src = "/home/ruiming/workspace/pro/source/yolov5"

sys.path.append(yolov5_src)

import argparse
import glob
from pathlib import Path
import cv2
import numpy as np

from smoke_keypoint_python import smoke_keypoint
from Facealign import FaceAlignment

from multiprocessing import Process, Lock, Value

def parse_args():
    parser = argparse.ArgumentParser(description='Build file list')
    parser.add_argument(
        'src_folder', type=str, help='root directory for the frames')
    parser.add_argument(
        'out_dir', type=str, help='out name')
    parser.add_argument(
        '--use_scrfd',
        action='store_true',
        default=False,
        help='choose face detection handler, yolov5 or scrfd')
    parser.add_argument(
        '--ext', type=str, default='jpg', help='out name')
    parser.add_argument(
        '--level',
        type=int,
        default=1,
        choices=[1, 2, 3],
        help='directory level of video dir')
    parser.add_argument(
        '--num-worker',
        type=int,
        default=10,
        help='num workers to preprocess')
    parser.add_argument(
        '--cpu',
        action='store_true',
        default=False,
        help='whether to use cpu')

    args = parser.parse_args()
    return args

def get_out_face_region(image, rect):
    sx, sy, ex, ey = rect
    h, w, c = image.shape
    faceh = ey - sy
    facew = ex - sx

    longsize = max(faceh, facew)
    longsize = longsize + longsize*0.4
    expendw = longsize - facew
    expendh = longsize - faceh

    sx = sx - (expendw / 2)
    ex = ex + (expendw / 2)
    sy = sy - (expendh / 2)
    ey = ey + (expendh / 2)

    sx = int(max(0, sx))
    sy = int(max(0, sy))
    ex = int(min(w - 1, ex))
    ey = int(min(h - 1, ey))

    return image[sy:ey, sx:ex, :], sx, sy, ex, ey

def get_smoking_rect_by_facialpoints(image, pts, facerect):
    shape = image.shape
    if len(shape) == 3:
        h,w,_ = shape
    else:
        h,w = shape
    # get mouth points (13_left, 14_up, 15_right, 16_down)
    mouth = pts[13:17]
    mouth = np.array(mouth)
    sx = min(mouth[:, 0])
    sy = min(mouth[:, 1])
    ex = max(mouth[:, 0])
    ey = max(mouth[:, 1])

    # border check
    msx = int(max(0, sx))
    msy = int(max(0, sy))
    mex = int(ex)
    mey = int(ey)

    cx = int(msx + (mex - msx) / 2)
    cy = int(msy + (mey - msy) / 2)

    # face rect
    ratio = 0.45
    face_sx, face_sy, face_ex, face_ey = facerect
    face_h = face_ey-face_sy
    face_w = face_ex-face_sx
    face_size = max(face_w, face_h)
    det_size = int(face_size*ratio)

    det_sx = max(0, cx - det_size)
    det_ex = min(w, cx + det_size)
    det_sy = max(0, cy - det_size)
    det_ey = min(h, cy + det_size)

    return cx, cy, det_sx, det_sy, det_ex, det_ey

def frame_deal(dirs, args, lock, counter, total_length):
    # init face detection model
    if args.use_scrfd:
        from utils.face_det_python import scrfd
        fd = scrfd.ScrdfFaceDet(0.45,
                                model_path='utils/face_det_python/models/model.pth',
                                device='cpu' if args.cpu else 'cuda',
                                config='utils/face_det_python/models/scrfd_500m.py')
    else:
        from FaceDetection import FaceDetect
        args.weights = os.path.join(yolov5_src, 'weights/200_last.pt')
        args.imgsz = 640
        fd = FaceDetect(args=args)
    # smoke point
    model_name = 'restiny_coco_128x128'
    epoch = 210
    sk = smoke_keypoint.SmokeKeypoint('../common_utils/smoke_keypoint_python/models/{}/{}.py'.format(model_name, model_name),
                                 '../common_utils/smoke_keypoint_python/models/{}/epoch_{}.pth'.format(model_name, epoch),
                                 device='cpu' if args.cpu else 'cuda',
                                      grayscale=True)
    # face alignment
    fa = FaceAlignment.faceAlignment('../common_utils/Facealign/models/multiScale_7_20210716.pkl')

    for vdir in dirs:
        imgs = glob.glob(os.path.join(vdir, '*.jpg'))
        for img in imgs:
            image = cv2.imread(img)
            
            if args.use_scrfd:
                result = fd.forward(image)
                if len(result) < 1:
                    continue
                box = result[0]
                sx, sy, ex, ey, prob = box
                if prob < 0.45:
                    continue
            else:
                bbox = fd.detect(image)
                if len(bbox) != 4 or sum(bbox) < 1:
                    print("cat not detect face")
                    continue
                else:
                    sx, sy, ex, ey = bbox

                # face align test
                eye, mouth = fa.forward_with_rect(image, (sx, sy, ex, ey))
                pts_pre = np.concatenate((eye, mouth), axis=0)

                font = cv2.FONT_HERSHEY_SIMPLEX

                # alignx, aligny, alignex, alighey = fa.det_area
                # cv2.rectangle(image, (int(alignx), int(aligny)), (int(alignex), int(alighey)), (0, 0, 255), 5)
                # for i in range(pts_pre.shape[0]):
                #     if i not in range(13, 17):
                #         continue
                #     cv2.circle(image, (int(pts_pre[i][0]), int(pts_pre[i][1])), 3, (255, 0, 0), -1)
                #     cv2.putText(image, str(i), (int(pts_pre[i][0]), int(pts_pre[i][1])), font, 0.4, (255, 255, 255), 1)

                # gen det center points
                #cx, cy, detsx, detsy, detex, detey = get_center_acc_facial_points(image, pts_pre)
                cx, cy, detsx, detsy, detex, detey = get_smoking_rect_by_facialpoints(image, pts_pre, (sx, sy, ex, ey))

                cv2.circle(image, (int(cx), int(cy)), 2, (164, 183, 214), 2)
                cv2.rectangle(image, (detsx, detsy), (detex, detey), (0, 255, 0), 10)

                points = sk(image, (detsx, detsy, detex, detey))
                color_map = {0: (0, 0, 255), 1: (0, 255, 0), 2: (255, 0, 0)}
                points_str = ''
                smoke_count = 0
                for idx, p in enumerate(points):
                    x, y, score = p
                    points_str += '{:.2f}  '.format(score)
                    if score < 0.7:
                        continue
                    smoke_count += 1

                    cv2.circle(image, (int(x), int(y)), 1, color_map[idx], 2)

                # debug message
                if smoke_count==3:
                    print(points_str)
                cv2.rectangle(image, (sx, sy), (ex, ey), (255, 0, 0), 10)
                sk_sx, sk_sy, sk_ex, sk_ey = sk.det_area
                cv2.rectangle(image, (sk_sx, sk_sy), (sk_ex, sk_ey), (255, 255, 0), 10)
                cv2.putText(image, points_str, (sx, sy - 20),
                            0, 2, (0, 0, 255), 2)
                image = cv2.resize(image, (int(image.shape[1]/2), int(image.shape[0]/2)))
                cv2.imshow('debug', image)
                if smoke_count == 3:
                    if cv2.waitKey(0) & 0xff == ord('q'):
                        break
                else:
                    if cv2.waitKey(1) & 0xff == ord('q'):
                        break

            # debug
            # cv2.rectangle(image, (fsx, fsy), (fex, fey), (255, 0, 0), 10)
            # cv2.putText(image, '{}:{:.3f}'.format(pred_name, pred_sclore), (sx, sy - 20),
            #             0, 2, (0, 0, 255), 2)
            # cv2.imshow('debug', image)
            # if cv2.waitKey(0) & 0xff == ord('q'):
            #     break

        # # counter
        # lock.acquire()
        # try:
        #     # p_bar.update(1)
        #     counter.value += 1
        #     if counter.value % 50 == 0:
        #         print(f"{counter.value}/{total_length} done.")
        # finally:
        #     lock.release()
    # debug
    cv2.destroyAllWindows()

def multi_process(frames_list, args):
    process_num = args.num_worker
    # check if is valid
    to_process_frames_list = [x for x in frames_list if os.path.isdir(x)]

    # start process
    files = to_process_frames_list
    grid_size = len(files) // process_num
    process_pool = []
    lock = Lock()
    counter = Value("i", 0)
    for i in range(process_num):
        start_index = grid_size * i
        if i != process_num - 1:
            end_index = grid_size * (i + 1)
        else:
            end_index = len(files)
        pw = Process(target=frame_deal,
                     args=(files[start_index:end_index], args, lock, counter, len(files)))
        pw.start()
        process_pool.append(pw)

    for p in process_pool:
        p.join()

def main():
    args = parse_args()

    # create out dir
    if not os.path.isdir(args.out_dir):
        os.makedirs(args.out_dir)

    # get all images folders
    frames_list = glob.glob(os.path.join(args.src_folder, str(Path('*/'*args.level))))

    # multi process
    multi_process(frames_list, args)

if __name__ == '__main__':
    main()

