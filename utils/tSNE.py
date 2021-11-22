"""Attempts to visualize highdim embeddings produced on the support images
by a few-shot trained network.
"""
import os
import h5py
import sys
import numpy as np
import json

from sklearn.manifold import TSNE
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns

def main(baseline='k-net', fname='snapshot-test-results.h5', basepath ='./object_reasoner/object_reasoner/data'):
    path_to_hdf5 = os.path.join(basepath,baseline, 'snapshots-with-class',fname)
    tgt_impr = h5py.File(path_to_hdf5, 'r')
    #Loading img embeddings
    emb_space = np.array(tgt_impr['prodFeat'], dtype='<f4') # 300 x 2048
    # Loading related labels and dictionaries to translate from numeric to human-readable labels
    with open(os.path.join(basepath,'Lab-set','test-product-labels.txt')) as prf,\
        open(os.path.join(basepath,'Lab-set', 'class_to_index.json')) as cin:
        plabels = prf.read().splitlines()
        mapper = json.load(cin)
    remapper = dict((v, k) for k, v in mapper.items())
    plabels_read = [remapper[l] for l in plabels]
    df = pd.DataFrame(emb_space)
    df['labels'] = plabels_read
    df['num_labels'] = [int(s) for s in plabels]

    #distribution estimated on whole set
    tsne = TSNE(n_components=2, verbose=1, metric='cosine', random_state=1, perplexity=50, n_iter=1000, learning_rate=300.)
    tsne_results = tsne.fit_transform(emb_space)
    df['tsne-2d-one'] = tsne_results[:, 0]
    df['tsne-2d-two'] = tsne_results[:, 1]

    # filter only classes from A to B for visualization
    A = 1
    B = 10
    dfsub = df.loc[(df['num_labels'] >= A) & (df['num_labels'] <= B)]

    plt.figure(figsize=(16, 10))
    sns.scatterplot(
        x="tsne-2d-one", y="tsne-2d-two",
        hue="labels",
        style="labels",
        s=150,
        palette=sns.color_palette("hls", B-A+1),
        data=dfsub,
        legend="full"
    )
    plt.legend(bbox_to_anchor=(1,1.25), loc='upper left', ncol=1)
    plt.show()
    return 0

if __name__ == '__main__':
    sys.exit(main())