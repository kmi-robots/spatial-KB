"""
Reasoner class

where knowledge-based correction of the ML predictions is applied
"""

import numpy as np
import json
import time
import networkx as nx
from evalscript import eval_singlemodel
from PostGIS import *


class ObjectReasoner():
    def __init__(self, args):
        with open(os.path.join(args.path_to_pred, 'test-labels.txt')) as txtf, \
                open(os.path.join(args.path_to_pred, 'class_to_index.json')) as cin, \
                open(os.path.join(args.path_to_pred, 'test-imgs.txt')) as txti:
            self.labels = txtf.read().splitlines()  # gt labels for each test img
            self.mapper = json.load(cin)
            self.fnames= [p.split('/')[-1].split('.')[0] for p in txti.read().splitlines()] #extract basename from imgpath
        self.predictions = np.load(('%s/test_predictions_%s.npy' % (args.path_to_pred, args.baseline)),
                                   allow_pickle=True)

    def run(self, eval_dictionary, spatialDB):
        """Similarly to the proposed size reasoner, we go image by image and find the ref-figure set,
         then in descending volume order, compute QSRs only for nearby objects to each"""

        """ Evaluate ML predictions before hybrid reasoning"""
        print("Evaluating ML baseline...")
        eval_dictionary = eval_singlemodel(self, eval_dictionary, 'MLonly')
        eval_dictionary = eval_singlemodel(self, eval_dictionary, 'MLonly', K=5)

        print("Reasoning for correction ... ")
        start = time.time()
        tmp_conn, tmp_cur = connect_DB(spatialDB.db_user, spatialDB.dbname) #open spatial DB connection
        already_processed = []
        QSRs = nx.MultiDiGraph() # all QSRs tracked in directed multi-graph (each node pair can have more than one connecting edge, edges are directed)
        for fname, pred_ranking, gt_label in list(zip(self.fnames, self.predictions, self.labels)):
            tstamp = '_'.join(fname.split('_')[:-1])
            if tstamp not in already_processed: #first time regions of that image are found.. extract all QSRs
                already_processed.append(tstamp)  # to skip other crops which are within the same frame
                all_ids = retrieve_ids_ord((tmp_conn,tmp_cur),tstamp) # find all other spatial regions at that timestamp in db
                QSRs.add_nodes_from(all_ids.keys())
                for o_id, _ in all_ids.items(): # find figures of each reference
                    figure_objs = find_neighbours((tmp_conn,tmp_cur), o_id, all_ids)
                    if len(figure_objs)>0: #, if any
                        #Find base QSRs between figure and nearby ref
                        QSRs = extract_QSR((tmp_conn,tmp_cur),o_id,figure_objs,QSRs)
                # after all references in image have been examined
                # derive special cases of ON
                img_QSRs = QSRs.subgraph(all_ids.keys())  # from global graph to local, i.e., only rel in that image
                QSRs = infer_special_ON(QSRs, img_QSRs)

            #Once all QSRs for one image have been extracted:
            # Lookup QSRs which involve current object region
            #TODO TBD, correct all crops in that image and indent the below? In that case we may skip global qsr graph
            # and just keep a local one
            # validate QSRs based on background knowledge
            #TODO go from QSRs between ids to QSRs between object labels
            #TODO handle/propose correction

        disconnect_DB(tmp_conn, tmp_cur) #close spatial DB connection
        procTime = float(time.time() - start)  # global proc time
        print("Took % fseconds." % procTime)
        eval_dictionary['spatial_VG']['processingTime'].append(procTime)
        return eval_dictionary

