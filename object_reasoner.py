"""
Reasoner class

where knowledge-based correction of the ML predictions is applied
"""

import numpy as np
import json
import time
import statistics
import networkx as nx
from evalscript import eval_singlemodel
from PostGIS import *
from utils import graphs as ugr


class ObjectReasoner():
    def __init__(self, args,idlist):
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
        self.filter_nulls(idlist)

    def filter_nulls(self,idlist):
        """remove filenames, labels and predictions which had been filtered from spatial DB
        """
        indices = [k for k,f in enumerate(self.fnames) if f in idlist]
        self.fnames = [self.fnames[i] for i in indices]
        self.labels = [self.labels[i] for i in indices]
        self.predictions = self.predictions[indices]

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
        walls = retrieve_walls(tmp_cur) #retrieve all walls of map first
        already_processed = []
        for fname in self.fnames:
            tstamp = '_'.join(fname.split('_')[:-1])
            if tstamp not in already_processed: #first time regions of that image are found.. extract all QSRs
                tstamp = '2020-05-15-11-02-54_646'  # TODO remove when running on full set
                QSRs = nx.MultiDiGraph() # all QSRs tracked in directed multi-graph (each node pair can have more than one connecting edge, edges are directed)
                already_processed.append(tstamp)  # to skip other crops which are within the same frame
                img_ids = retrieve_ids_ord((tmp_conn,tmp_cur),tstamp) # find all other spatial regions at that timestamp in db
                QSRs.add_nodes_from(img_ids.keys())
                lmapping = dict((o_id, str(i) + '_' + self.remapper[self.labels[self.fnames.index(o_id)]]) for i,o_id in enumerate(img_ids.keys()))
                lmapping['floor'] ='floor'
                lmapping['wall'] ='wall'
                for i, o_id in enumerate(img_ids.keys()): # find figures of each reference
                    figure_objs = find_neighbours((tmp_conn,tmp_cur), o_id, img_ids)
                    if len(figure_objs)>0: #, if any
                        #Find base QSRs between figure and nearby ref
                        QSRs = extract_QSR((tmp_conn,tmp_cur),o_id,figure_objs,QSRs)
                        QSRs = extract_surface_QSR((tmp_conn,tmp_cur),o_id,walls,QSRs)

                # after all references in image have been examined
                # derive special cases of ON
                QSRs = infer_special_ON(QSRs)
                QSRs_H = nx.relabel_nodes(QSRs,lmapping) #human-readable ver
                ugr.plot_graph(QSRs_H)

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

    def space_validate(self,obj_list,qsr_graph,spatialDB, K=10):
        # correction independent from the order objects in each img are picked, order of ML ranking and then combine topK scores

        for oid in obj_list: #for each object to correct/validate
            i = self.fnames.index(oid)
            ML_rank = self.predictions[i, :]
            hybrid_rank_atK = np.copy(ML_rank[:K])
            for n, (cnum, L2dis) in enumerate(ML_rank[:K]): #for each class in the top-K rank
                pred_label = self.remapper[cnum]
                wn_syn = self.taxonomy[pred_label] #wordnet synset for that label
                if not wn_syn: continue  # skip objects that do not have a mapping to VG through WN (foosball table and pigeon holes)

                # retrieve QSRs for that object
                fig_qsrs = [(pred_label,self.remapper[self.labels[self.fnames.index(ref)]],r['QSR'])
                        for f,ref,r in qsr_graph.out_edges(oid, data=True) if ref not in ['wall','floor']] #rels where obj is figure
                ref_qsrs = [(self.remapper[self.labels[self.fnames.index(f)]],pred_label,r['QSR'])
                        for f,ref,r in qsr_graph.in_edges(oid, data=True) if f not in ['wall','floor']] # rels where obj is reference

                #Retrieve wall and floor QSRs, only in figure/reference form - e.g., 'object onTopOf floor'
                surface_qsrs = [(pred_label,ref,r['QSR']) for f,ref,r \
                                in qsr_graph.out_edges(oid, data=True) if ref in ['wall','floor']]
                fig_qsrs.extend(surface_qsrs) # merge into list of fig/ref relations
                #Tipicality scores based on VG stats
                sub_syn = self.taxonomy[pred_label]
                all_spatial_scores = []
                for _,ref,r in fig_qsrs: #for each QSR where obj is figure, i.e., subject
                    if ref=='wall': obj_syn = 'wall.n.01' #cases where reference is wall or floor
                    elif ref=='floor': obj_syn = 'floor.n.01'
                    else: obj_syn = self.taxonomy[ref]
                    if len(obj_syn)>1: # more than one synset
                        typscores =[self.compute_typicality_score(spatialDB,sub_syn,osyn,r) for osyn in obj_syn]
                        typscores = [s for s in typscores if s != 0.]  # keep only synset that of no-null typicality
                                                                       # in order of taxonomy (from preferred synset to least preferred)
                        if len(typscores) == 0:
                            typscore = 0.
                        else: typscore = typscores[0]  # first one in the order
                    else: typscore = self.compute_typicality_score(spatialDB,sub_syn,obj_syn,r)
                    all_spatial_scores.append((1. - typscore))  #track INVERSE of score (so that it is comparable
                                                                # with L2 distances, i.e., scores that are minimised)

                # Similarly, for QSRs where predicted obj is reference, i.e., object
                obj_syn = self.taxonomy[pred_label]
                for fig,_,r in ref_qsrs:
                    sub_syn = self.taxonomy[fig]
                    if len(obj_syn) > 1:
                        typscores = [self.compute_typicality_score(spatialDB, ssyn, obj_syn, r) for ssyn in sub_syn]
                        typscores = [s for s in typscores if s!= 0.] # keep only synset that of no-null typicality in order of taxonomy (from preferred synset to least preferred)
                        if len(typscores)==0: typscore = 0.
                        else: typscore = typscores[0] #first one in the order
                    else: typscore = self.compute_typicality_score(spatialDB, sub_syn, obj_syn, r)
                    all_spatial_scores.append((1. - typscore))  # track INVERSE of score

                # Average across all QSRs
                avg_spatial_score = statistics.mean(all_spatial_scores)
                hybrid_rank_atK[n][1] += avg_spatial_score # add up to ML score

            # Normalise scores across the topK classes, so it is between 0 and 1
            # minmax norm
            scores = hybrid_rank_atK[:,1]
            min_, max_ = np.min(scores), np.max(scores)
            hybrid_rank_atK[:, 1] = np.array([(x-min_)/(max_ - min_) for x in scores])
            posthoc_rank = hybrid_rank_atK[np.argsort(hybrid_rank_atK[:, 1])] # order by score ascending
            # ranking after correction is ..
            # replace predictions at that index with corrected predictions
            self.predictions[i, :] = posthoc_rank

    def compute_typicality_score(self,spatialDB,sub_syn,obj_syn,rel):
        #no of times the two appeared in that relation in VG
        try:
            nom = float(spatialDB.KB.VG_stats['predicates'][rel]['relations'][str((str(sub_syn), str(obj_syn)))])
        except KeyError: #if any hit is found
            return 0.
        #no of times sub_syn was subject of r relation in VG
        try:
            denom1 = float(spatialDB.KB.VG_stats['subjects'][rel][sub_syn])
        except KeyError: #if any hit is found
            return 0.
        #no of times obj_syn was object of r relation in VG
        try:
            denom2 = float(spatialDB.KB.VG_stats['objects'][rel][obj_syn])
        except KeyError: #if any hit is found
            return 0.
        return nom / (denom1+denom2)