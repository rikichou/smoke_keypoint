import sys
import os
import json
import numpy as np

NEG_ANNS_PATH = r'E:\workspace\pro\smoke_keypoint\data\test\search_20220104_negative.json'
POS_ANNS_PATH = r'E:\workspace\pro\smoke_keypoint\data\test\search_20220104_smoke.json'

OUT_JSON_PATH = r'E:\workspace\pro\smoke_keypoint\data\test\results_20220104.json'

def get_json_infos(anns_path, label):
    vnames = []
    scores = []
    labels = []
    
    with open(anns_path, 'r') as fp:
        infos = json.load(fp)

    for name in infos:
        vnames.append(name)
        scores.append(infos[name])
        labels.append(label)
    
    return vnames, scores, labels

neg_vnames, neg_scores, neg_labels = get_json_infos(NEG_ANNS_PATH, 0)
pos_vnames, pos_scores, pos_labels = get_json_infos(POS_ANNS_PATH, 1)

neg_vnames.extend(pos_vnames)
vnames = neg_vnames

neg_scores.extend(pos_scores)
scores = neg_scores

neg_labels.extend(pos_labels)
labels = neg_labels

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

    # get positive acc
    mask = labels==1
    pos_acc = sum(labels[mask]==pred[mask])/len(pred[mask])
    
    # get negative acc
    mask = labels==0
    neg_acc = sum(labels[mask]==pred[mask])/len(pred[mask])

    return acc, pos_acc, neg_acc

#print(get_accuracy(scores, labels, [0.8, 0.8, 0.8]))

accs = []
pos_accs = []
neg_accs = []
threads = []
for t1 in np.arange(0.7,0.9,0.05):
    for t2 in np.arange(0.7,0.9,0.05):
        for t3 in np.arange(0.7,0.9,0.05):
            accuracy, pos_acc, neg_acc = get_accuracy(scores, labels, [t1, t2, t3])
            
            accs.append(accuracy)
            pos_accs.append(pos_acc)
            neg_accs.append(neg_acc)
            
            threads.append([t1, t2, t3])
            print(t1, t2, t3)
out_info = {}
out_info['accuracy'] = accs
out_info['pos_accs'] = pos_accs
out_info['neg_accs'] = neg_accs
out_info['threads'] = threads

with open(OUT_JSON_PATH, 'w') as fp:
    json.dump(out_info, fp)



