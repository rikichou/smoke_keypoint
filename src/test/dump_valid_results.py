import cv2
import numpy as np
import json

OUT_JSON_PATH = r'E:\workspace\pro\smoke_keypoint\data\test\results_20220104.json'

with open(OUT_JSON_PATH, 'r') as fp:
    infos = json.load(fp)

accs = np.array(infos['accuracy'])
pos_accs = np.array(infos['pos_accs'])
neg_accs = np.array(infos['neg_accs'])
threads = np.array(infos['threads'])

max_idx = np.argmax(accs)
print("Max Accuracy:{}, pos_acc:{}, neg_acc{}, thread:{}".format(accs[max_idx], pos_accs[max_idx], neg_accs[max_idx], threads[max_idx]))

max_idx = np.argmax(pos_accs)
print("Accuracy:{}, Max pos_acc:{}, neg_acc{}, thread:{}".format(accs[max_idx], pos_accs[max_idx], neg_accs[max_idx], threads[max_idx]))

max_idx = np.argmax(neg_accs)
print("Accuracy:{}, pos_acc:{}, Max neg_acc{}, thread:{}".format(accs[max_idx], pos_accs[max_idx], neg_accs[max_idx], threads[max_idx]))