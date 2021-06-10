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
                open(os.path.join(args.path_to_pred, 'class_to_synset.json')) as sin, \
                open(os.path.join(args.path_to_pred, 'test-imgs.txt')) as txti:
            self.labels = txtf.read().splitlines()  # gt labels for each test img
            self.mapper = json.load(cin)
            self.fnames= [p.split('/')[-1].split('.')[0] for p in txti.read().splitlines()] #extract basename from imgpath
            self.taxonomy = json.load(sin)
        self.predictions = np.load(('%s/test_predictions_%s.npy' % (args.path_to_pred, args.baseline)),
                                   allow_pickle=True)
        self.remapper = dict((v, k) for k, v in self.mapper.items())  # swap keys with indices
        self.scenario = args.scenario

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
        for fname in self.fnames:
            tstamp = '_'.join(fname.split('_')[:-1])
            if tstamp not in already_processed: #first time regions of that image are found.. extract all QSRs
                QSRs = nx.MultiDiGraph() # graph is local to the img and thrown away afterwards to speed up # all QSRs tracked in directed multi-graph (each node pair can have more than one connecting edge, edges are directed)
                already_processed.append(tstamp)  # to skip other crops which are within the same frame
                img_ids = retrieve_ids_ord((tmp_conn,tmp_cur),tstamp) # find all other spatial regions at that timestamp in db
                QSRs.add_nodes_from(img_ids.keys())
                Gt_ind, Gt_labs =[], []
                for o_id, _ in img_ids.items(): # find figures of each reference
                    Gt_ind.append(self.fnames.index(o_id))
                    Gt_labs.append(self.remapper[self.labels[self.fnames.index(o_id)]])
                    figure_objs = find_neighbours((tmp_conn,tmp_cur), o_id, img_ids)
                    if len(figure_objs)>0: #, if any
                        #Find base QSRs between figure and nearby ref
                        QSRs = extract_QSR((tmp_conn,tmp_cur),o_id,figure_objs,QSRs)
                # after all references in image have been examined
                # derive special cases of ON
                QSRs = infer_special_ON(QSRs)

                # which ML predictions to correct in that image?
                if self.scenario =='best':
                    #correct only ML predictions which need correction, i.e., where ML prediction differs from ground truth
                    tbcorr = [id_ for id_ in img_ids if self.labels[self.fnames.index(id_)] != \
                              self.predictions[self.fnames.index(id_),0,0] ]
                elif self.scenario == 'selected': # select for correction, based on confidence
                    tbcorr = [id_ for id_ in img_ids if self.predictions[self.fnames.index(id_), 0, 1] \
                              >= self.epsilon_set[0]] #where L2 distance greater than conf thresh
                else: tbcorr = img_ids #validate all

                # proceed with validation/correction based on spatial knowledge
                self.space_validate(tbcorr, QSRs,spatialDB)
                #TODO integrate size correction as well,
                # but this time dimensions are derived from postgis database
                # and image-wise instead of crop by crop
                # Note: imgs with empty pcls or not enough points were skipped in prior size reasoning exps

        disconnect_DB(tmp_conn, tmp_cur) #close spatial DB connection
        procTime = float(time.time() - start)  # global proc time
        print("Took % fseconds." % procTime)
        eval_dictionary['spatial_VG']['processingTime'].append(procTime)

        #Re-eval post correction
        print("Hybrid results (spatial-only)")
        eval_dictionary = eval_singlemodel(self, eval_dictionary, 'spatial_VG')
        eval_dictionary = eval_singlemodel(self, eval_dictionary, 'spatial_VG', K=5)
        return eval_dictionary

    def space_validate(self,obj_list,qsr_graph,spatialDB):
        all_classes = list(self.mapper.keys())
        # To make correction independent from the order objects in each img are picked, we interpret QSRs based on original ML prediction and do not correct labels in the QSR

        for oid in obj_list: #for each object to correct/validate
            i = self.fnames.index(oid)
            ML_rank = self.predictions[i, :]
            label_txt = self.remapper[self.labels[i]]
            pred_label = self.remapper[ML_rank[0][0]]

            # retrieve QSRs for that object
            fig_qsrs = [(label_txt,self.remapper[self.labels[self.fnames.index(ref)]],r['QSR'])
                        for f,ref,r in qsr_graph.out_edges(oid, data=True)] #rels where obj is figure
            ref_qsrs = [(self.remapper[self.labels[self.fnames.index(f)]],label_txt,r['QSR'])
                        for f,ref,r in qsr_graph.in_edges(oid, data=True)] # rels where obj is reference

            #Class tipicality scores based on VG stats
            """sub_syn, obj_syn = None, None
            for _,ref,r in fig_qsrs: #rels where obj is figure

                fprobs = [(c,float(spatialDB.KB.VG_stats['predicates']\
                                       [r]['relations'][str((str(sub_syn), str(obj_syn)))] \
                                   /spatialDB.KB.VG_stats['objects'][str(obj_syn)][r] )) \
                                        for c in all_classes]
                fprobs =sorted(fprobs, key=lambda x: x[1], reverse=True) #sort by prob, descending
                #TODO combine across relations/QSRs

            for f,_,r in ref_qsrs: # rels where obj is reference
                refprobs = [(c,float(spatialDB.KB.VG_stats['predicates']\
                                       [r]['relations'][str((str(sub_syn[0]), str(obj_syn[0])))] \
                                   /spatialDB.KB.VG_stats['subjects'][str(obj_syn[0])][r] )) \
                                        for c in all_classes]
                refprobs =sorted(refprobs, key=lambda x: x[1], reverse=True) #sort by prob, descending
                # TODO combine across relations/QSRs

            #TODO combine fprobs and refprobs together
            # + weight all by ML confidence? (inverse of L2 dis in ranking)
            """
        return