import sys
import os
import json
import numpy as np
import glob

NEG_IMGS_DIR = r'E:\workspace\pro\smoke_keypoint\data\test\images\negative'
POS_IMGS_DIR = r'E:\workspace\pro\smoke_keypoint\data\test\images\positive'

OUT_JSON_PATH = r'E:\workspace\pro\smoke_keypoint\data\test\images\results_images.json'

def get_images_infos(images_dir, label):
    scores = []
    labels = []

    images_paths = glob.glob(os.path.join(images_dir, r'*\*.jpg'))

    for img_path in images_paths:
        # get scores img_{:05d}_{:.2f}_{:.2f}_{:.2f}.jpg
        name = os.path.basename(img_path)
        score = os.path.splitext(name)[0].split('_')[2:]
        score = [float(num) for num in score]
        scores.append(score)
        # get labels
        labels.append(label)
    
    return scores, labels

neg_scores, neg_labels = get_images_infos(NEG_IMGS_DIR, 0)
pos_scores, pos_labels = get_images_infos(POS_IMGS_DIR, 1)
#pos_scores = []
#pos_labels = []

neg_scores.extend(pos_scores)
scores = neg_scores

neg_labels.extend(pos_labels)
labels = neg_labels

def get_predict(scores, threaholds:list):
    predict = []
    threaholds = np.array(threaholds)
    # get videos scores
    v_score = scores
    v_score = np.array(v_score)
    c = v_score>=threaholds
    
    res = np.sum(c, axis=1)
    res = res==3
    
    predict = np.zeros(len(res))
    predict[res] = 1

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
    #pos_acc = 0
    
    # get negative acc
    mask = labels==0
    neg_acc = sum(labels[mask]==pred[mask])/len(pred[mask])

    return acc, pos_acc, neg_acc

print(get_accuracy(scores, labels, [0.8, 0.8, 0.8]))

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



