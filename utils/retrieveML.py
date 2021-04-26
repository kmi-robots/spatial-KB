"""
Update PostGRE SQL table with
predictions generated through Machine Learning
"""

import sys
import os
import json
import numpy as np
from PostGIS import connect_DB, disconnect_DB


def load_preds(path_to_npy):
    return np.load(path_to_npy, allow_pickle=True)


def load_txt_files(basepath):
    with open(os.path.join(basepath, 'test-labels.txt')) as txtf, \
            open(os.path.join(basepath, 'test-imgs.txt')) as imgf:
        labels = txtf.read().splitlines()  # gt labels for each test img
        imglist = imgf.read().splitlines()
    return labels,imglist

def load_class_index(path_to_json):
    with open(path_to_json) as fin:
        mapper = json.load(fin)
    return dict((v, k) for k, v in mapper.items())  # swap keys with indices

def main():
    MLpredictions = load_preds('./data/logged-predictions/test_predictions_k-net.npy')
    # dictionary to convert numeric labels to object names
    remapper = load_class_index('./data/logged-predictions/class_to_index.json')
    gt_labels, img_list = load_txt_files('./data/logged-predictions')
    conn, cur = connect_DB(os.environ['USER'], 'VG_spatial')
    # name of data table to update

    if conn is not None:
        for i,filepath in enumerate(img_list):
            object_id = filepath.split('/')[-1][:-4]
            label_ranking = [remapper[n] for n in MLpredictions[i,:5,0]]
            score_ranking = MLpredictions[i,:5,1].tolist()
            top1_label = remapper[MLpredictions[i,0,0]]
            top1_score = MLpredictions[i,0,1]
            gt_label = remapper[gt_labels[i]]

            cur.execute("""
                        UPDATE semantic_map
                        SET ML_prediction = %s,
                        ML_score = %s,
                        gt_label = %s,
                        top5_labels = %s,
                        top5_scores = %s
                        WHERE object_id = %s;
            """,(top1_label,top1_score, gt_label, (label_ranking,), (score_ranking,),object_id))

        # commit all changes and close connection with DB
        conn.commit()
        disconnect_DB(conn, cur)



if __name__ == '__main__':
    sys.exit(main())
