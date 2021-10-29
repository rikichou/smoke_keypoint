import os
import json

src_json_file = '/Volumes/10.20.132.160-1/pro/smoke_keypoint/data/smoke_keypoint_mpii/annotations/all.json'
train = '/Volumes/10.20.132.160-1/pro/smoke_keypoint/data/smoke_keypoint_mpii/annotations/train.json'
val = '/Volumes/10.20.132.160-1/pro/smoke_keypoint/data/smoke_keypoint_mpii/annotations/val.json'

valid_num = 1000
withhand_num = valid_num/2
nohand_num = withhand_num

withhand_count = 0
nohand_count = 0
train_count = 0

with open(src_json_file, 'r') as fp:
    anns = json.load(fp)

train_ann = []
val_ann = []

for ann in anns:
    point_type = ann['point_type']

    # check if we have enough info
    if point_type == 'with_hand' and withhand_count != withhand_num:
        withhand_count += 1
        val_ann.append(ann)
    elif point_type == 'no_hand' and nohand_count != nohand_num:
        nohand_count += 1
        val_ann.append(ann)
    else:
        train_count += 1
        train_ann.append(ann)

print("Get val {}, withhand {}, nohand {}, train {}".format(len(val_ann), withhand_count, nohand_count, train_count))

with open(train, 'w') as fp:
    json.dump(train_ann, fp)

with open(val, 'w') as fp:
    json.dump(val_ann, fp)