"""
Reasoner class

where knowledge-based correction of the ML predictions is applied
"""

import numpy as np
import json
import os
import time
from evalscript import eval_singlemodel


class ObjectReasoner():
    def __init__(self, args):

        with open(os.path.join(args.path_to_pred, 'test-labels.txt')) as txtf,\
            open(os.path.join(args.path_to_pred, 'class_to_index.json')) as cin:
            self.labels = txtf.read().splitlines()       #gt labels for each test img
            self.mapper = json.load(cin)
        self.predictions = np.load(('%s/test_predictions_%s.npy' % (args.path_to_pred, args.baseline)), allow_pickle=True)

    def run(self, eval_dictionary):
        """Similarly to the proposed size reasoner, we go image by image and find the ref-figure set,
         then in descending volume order, compute QSRs only for nearby objects to each"""

        """ Evaluate ML predictions before hybrid reasoning"""
        print("Evaluating ML baseline...")
        eval_dictionary = eval_singlemodel(self, eval_dictionary, 'MLonly')
        eval_dictionary = eval_singlemodel(self, eval_dictionary, 'MLonly', K=5)

        print("Reasoning for correction ... ")
        """Data stats and monitoring vars for debugging"""
        start = time.time()

        procTime = float(time.time() - start) # global proc time
        print("Took % fseconds." % procTime)
        eval_dictionary['spatial_VG']['processingTime'].append(procTime)

        return eval_dictionary




