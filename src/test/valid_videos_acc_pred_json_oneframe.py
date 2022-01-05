import sys
import os
import json
import numpy as np

vnames = []
scores = []
labels = []

NEGA_ANNS_PATH = r'E:\workspace\pro\smoke_keypoint\data\test\shanghai.json'
with open(NEGA_ANNS_PATH, 'r') as fp:
    nega_infos = json.load(fp)

for name in nega_infos:
    vnames.append(name)
    scores.append(nega_infos[name])
    labels.append(0)

def get_predict(scores, threaholds:list):
    predict = []
    threaholds = np.array(threaholds)
    # get videos scores
    for v_score in scores:
        v_score = np.array(v_score)
        c = v_score>=threaholds
        
        res = np.sum(c, axis=1)
        res = res==3
        
        if sum(res) >= 1:
            predict.append(1)
        else:
            predict.append(0)
        # ret = np.zeros(len(res))
        # ret[res] = 1
    return predict

def get_accuracy(scores, labels, threaholds:list):
    # get predict
    pred = get_predict(scores, threaholds)

    # get accuracy
    labels = np.array(labels)
    pred = np.array(pred)

    acc = sum(labels==pred)/len(pred)

    return acc

#print(get_accuracy(scores, labels, [0.8, 0.8, 0.8]))

accs = []
threads = []
for t1 in np.arange(0.7,1,0.02):
    for t2 in np.arange(0.7,1,0.02):
        for t3 in np.arange(0.7,1,0.02):
            accuracy = get_accuracy(scores, labels, [t1, t2, t3])
            accs.append(accuracy)
            threads.append([t1, t2, t3])
            print(t1, t2, t3)
print(accs)

