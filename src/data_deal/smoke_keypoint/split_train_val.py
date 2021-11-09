import os
import json

anns_dir = r'H:\pro\smoke_keypoint\data\smoke_keypoint_mpii_relabel\annotations'
positive_src_json_file = os.path.join(anns_dir, 'positive.json')
negative_src_json_file = os.path.join(anns_dir, 'negative.json')
train = os.path.join(anns_dir, 'train.json')
val = os.path.join(anns_dir, 'val.json')

valid_num = 1000
withhand_num = 250
nohand_num = withhand_num
negative_num = 500

withhand_count = 0
nohand_count = 0
negative_count = 0
train_count = 0

with open(positive_src_json_file, 'r') as fp:
    pos_anns = json.load(fp)
with open(negative_src_json_file, 'r') as fp:
    neg_anns = json.load(fp)

anns = pos_anns + neg_anns

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
    elif point_type == 'negative' and negative_count != negative_num:
        negative_count += 1
        val_ann.append(ann)
    else:
        train_count += 1
        train_ann.append(ann)

print("Get val {}, withhand {}, nohand {}, negative {}, train {}".format(len(val_ann), withhand_count, nohand_count, negative_count, train_count))

with open(train, 'w') as fp:
    json.dump(train_ann, fp)

with open(val, 'w') as fp:
    json.dump(val_ann, fp)