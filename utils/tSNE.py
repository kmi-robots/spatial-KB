"""Attempts to visualize highdim embeddings produced on the support images
by a few-shot trained network.
"""
import os
import h5py
import sys
import numpy as np

from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns

def main(baseline='k-net', fname='snapshot-test-results.h5'):
    path_to_hdf5 = os.path.join('./object_reasoner/object_reasoner/data',baseline, 'snapshots-with-class',fname)
    tgt_impr = h5py.File(path_to_hdf5, 'r')
    emb_space = np.array(tgt_impr['prodFeat'], dtype='<f4') # 300 x 2048

    return 0

if __name__ == '__main__':
    sys.exit(main())