"""
Reasoner class

where knowledge-based correction of the ML predictions is applied
"""

import numpy as np
import json
import time
import statistics
import networkx as nx
import cv2
from collections import OrderedDict, Counter
import random

from evalscript import eval_singlemodel
from PostGIS import *
from quantizer import quantize
# from utils import graphs as ugr


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
        with open(os.path.join(args.path_to_pred, 'test-imgs.txt')) as txti:
            lines = txti.read().splitlines()
        self.crops = [os.path.join('/'.join(p.split('/')[:-1]), ('_').join(rgbname.split('_')[:-1])+'depth_'+rgbname.split('_')[-1]+'.png')\
                          for p, rgbname in zip(lines, self.fnames)]  # extract width and height of each 2D obj crop
        self.crops = [cv2.imread(p).shape[:2] for p in self.crops]

        self.predictions = np.load(('%s/test_predictions_%s.npy' % (args.path_to_pred, args.baseline)),
                                   allow_pickle=True)
        self.remapper = dict((v, k) for k, v in self.mapper.items())  # swap keys with indices
        self.scenario = args.scenario
        self.reasoner_type = args.rm
        self.spatial_label_type = args.ql
        self.withML = args.withML
        self.waterfall = args.waterfall
        #filter out objects with no depth data associated in the spatial DB
        self.filter_nulls(idlist)

        #size reasoning params (autom derived from prior sorting of objects into histograms)
        self.T = [-4.149075426919093, -2.776689935975939, -1.4043044450327855, -0.0319189540896323]
        self.lam = [-2.0244465762356794, -1.0759355070093815, -0.12742443778308354]

    def filter_nulls(self,idlist):
        """remove filenames, labels and predictions which had been filtered from spatial DB
        """
        indices = [k for k,f in enumerate(self.fnames) if f in idlist]
        self.fnames = [self.fnames[i] for i in indices]
        self.labels = [self.labels[i] for i in indices]
        self.crops = [self.crops[i] for i in indices]
        self.predictions = self.predictions[indices]
        self.fnames_full = self.fnames #these full lists will not be subsampled in crossval, i.e., used to extract all QSRs nonetheless
        self.labels_full = self.labels
        self.pred_full = self.predictions
        self.crops_full = self.crops

    def run(self, eval_dictionary, spatialDB, sizeKB):
        """Similarly to the proposed size reasoner, we go image by image and find the ref-figure set,
         then in descending volume order, compute QSRs only for nearby objects to each"""

        """ Evaluate ML predictions before hybrid reasoning"""
        print("Evaluating ML baseline...")
        eval_dictionary = eval_singlemodel(self, eval_dictionary, 'MLonly')
        eval_dictionary = eval_singlemodel(self, eval_dictionary, 'MLonly', K=5)

        print("Reasoning for correction ... ")
        start = time.time()
        tmp_conn, tmp_cur = connect_DB(spatialDB.db_user, spatialDB.dbname) #open spatial DB connection
        disconnect_DB(tmp_conn, tmp_cur)  # close spatial DB connection
        sizeranks = self.predictions[:, :5, :] #set of only top-5 predictions for each object
        sizequal_copy = sizeranks.copy() #copies for ablation study on individual size features
        flat_copy = sizeranks.copy()
        thin_copy = sizeranks.copy()
        flatAR_copy = sizeranks.copy()
        thinAR_copy = sizeranks.copy()
        already_processed = []

        for fname in self.fnames:

            tstamp = '_'.join(fname.split('_')[:-1])
            if tstamp not in already_processed: #first time regions of that image are found.. extract all QSRs
                #tstamp = '2020-05-15-11-03-55_874916' #'2020-05-15-11-02-54_646666'  # test/debug/single_snap image
                print("============================================")
                print("Processing img %s" % tstamp)
                # for debugging only: visualize img
                """import cv2
                cv2.imshow('win', cv2.imread(os.path.join('/home/agnese/bags/KMi-set-new/' \
                                                          'test/rgb', tstamp + '.jpg')))
                cv2.waitKey(1000)
                cv2.destroyAllWindows()"""

                #QSRs are extracted for all img regions (self.fnames_full, self.labels_full)
                QSRs = nx.MultiDiGraph() # all QSRs tracked in directed multi-graph (each node pair can have more than one connecting edge, edges are directed)
                already_processed.append(tstamp)  # to skip other crops which are within the same frame

                tmp_conn, tmp_cur = connect_DB(spatialDB.db_user, spatialDB.dbname)  # open spatial DB connection
                img_ids = retrieve_ids_ord((tmp_conn,tmp_cur),tstamp) # find all other spatial regions at that timestamp in db
                # disconnect_DB(tmp_conn, tmp_cur)  # close spatial DB connection

                img_ids = OrderedDict({key_:vol for key_,vol in img_ids.items() if key_ in self.fnames_full})#exclude those that were filtered as null
                #subids = [self.fnames.index(key_) for key_ in img_ids.keys() if key_ in self.fnames]
                subimg_ids = OrderedDict({key_: vol for key_, vol in img_ids.items() if
                                          key_ in self.fnames})  # only those which are also in the subsampled set of folds are kept

                """Selecting what to correct"""
                # which ML predictions to correct in that image?
                # Correction is applied only for img regions in subsampled fold (self.fnames, self.labels)
                if self.scenario == 'best':
                    # correct only ML predictions which need correction, i.e., where ML prediction differs from ground truth
                    tbcorr = [id_ for id_ in subimg_ids if self.labels[self.fnames.index(id_)] != \
                              self.predictions[self.fnames.index(id_), 0, 0]]
                    QSRcandidates = img_ids # all count towards QSRs
                elif self.scenario == 'selected':
                    tbcorr = [id_ for id_ in subimg_ids if self.predictions[self.fnames.index(id_), 0, 1] \
                              >= self.epsilon_set[0]] # # select for correction, based on confidence, i.e., L2 distance greater than threshold
                    neg_tbcorr = [id_ for id_ in subimg_ids if self.predictions[self.fnames.index(id_), 0, 1] \
                              < self.epsilon_set[0]]
                    # QSRcandidates = img_ids
                    if self.reasoner_type =='size_spatial':
                        tbcorr, QSRcandidates = self.size_select(list(subimg_ids.keys()), list(img_ids.keys()), neg_tbcorr, sizeKB, (tmp_conn,tmp_cur)) # check which ones have a top-1 prediction which is valid wrt size
                    else: QSRcandidates = img_ids
                else:
                    tbcorr = img_ids  # validate all
                    QSRcandidates = img_ids

                if len(tbcorr)>0: #do reasoning/computation only if correction needed
                    print("Correcting objects which are not valid wrt size")
                    """Qualitative Size Reasoning"""
                    if self.reasoner_type =='random':
                        #simply changes ranking at random, with the option to repeat the same class name more than once
                        all_classes= list(self.remapper.keys())
                        dim =self.predictions.shape[1]
                        for oid in tbcorr:
                            ind = self.fnames.index(oid)
                            random_labels = [str(random.randint(1, len(all_classes))) for x in range(dim)]
                            self.predictions[ind,:,0] = np.array(random_labels)

                    if 'size' in self.reasoner_type:
                        for oid in tbcorr:
                            ind = self.fnames.index(oid)
                            d1,d2,d3 = extract_size((tmp_conn,tmp_cur),oid)# extract observed sizes based on dimensions of bbox on spatial DB
                            # print("Estimated dims oriented %f x %f x %f m" % (d1, d2, d3))
                            cropimg_shape = self.crops[self.fnames.index(oid)]
                            sres = self.size_validate_ranking([d1,d2,d3], self.lam, self.T, sizeKB, cropimg_shape)
                            candidates_num, candidates_num_flat, candidates_num_thin, candidates_num_flatAR, candidates_num_thinAR = sres

                            if self.withML:
                                # Keep only ML predictions which are plausible wrt size
                                full_vision_rank = self.predictions[self.fnames.index(oid)]
                                valid_rank_flatAR = full_vision_rank[[full_vision_rank[z, 0] in candidates_num_flatAR for z in range(full_vision_rank.shape[0])]]
                                valid_rank_thinAR = full_vision_rank[[full_vision_rank[z, 0] in candidates_num_thinAR for z in range(full_vision_rank.shape[0])]]

                                valid_rank = full_vision_rank[[full_vision_rank[z, 0] in candidates_num for z in range(full_vision_rank.shape[0])]]
                                valid_rank_flat = full_vision_rank[[full_vision_rank[z, 0] in candidates_num_flat for z in range(full_vision_rank.shape[0])]]
                                valid_rank_thin = full_vision_rank[[full_vision_rank[z, 0] in candidates_num_thin for z in range(full_vision_rank.shape[0])]]

                                # convert rankings to readable labels
                                read_res = self.makereadable(full_vision_rank, valid_rank, valid_rank_flat, valid_rank_thin,
                                                             valid_rank_flatAR, valid_rank_thinAR)
                                read_rank_ML, read_rank_area, read_rank_flat, read_rank_thin, read_rank_flatAR, read_rank_thinAR = read_res

                                # Verbose result printing for inspection
                                print("Initial ML rank")
                                print(read_rank_ML[:5])
                                print("Knowledge validated ranking (area)")
                                print(read_rank_area[:5])
                                #print("Knowledge validated ranking (area + flat)")
                                #print(read_rank_flat[:5])
                                #print("Knowledge validated ranking (area + thin)")
                                #print(read_rank_thin[:5])
                                if candidates_num_flatAR is not None:
                                    print("Knowledge validated ranking (area + flat + AR)")
                                    print(read_rank_flatAR[:5])
                                if candidates_num_thinAR is not None:
                                    print("Knowledge validated ranking (area + thin + AR)")
                                    print(read_rank_thinAR[:5])

                                if len(valid_rank_thinAR) > 0:
                                    #sizeranks[ind, :] = valid_rank_thinAR[:5, :]
                                    thinAR_copy[ind, :] = valid_rank_thinAR[:5, :]
                                if len(valid_rank_flatAR) > 0:
                                    flatAR_copy[ind, :] = valid_rank_flatAR[:5, :]
                                thin_copy[ind, :] = valid_rank_thin[:5, :]
                                sizequal_copy[ind, :] = valid_rank[:5, :]  # _thin[:5,:]
                                flat_copy[ind, :] = valid_rank_flat[:5, :]
                                #changed, keep size and ML predictions separate, unless, size only
                                # self.predictions[ind, :5] = sizeranks[ind, :] # change ML predictions
                                if self.reasoner_type=='size' or self.waterfall:
                                    # if size_only or size and spatial applied in sequence, change ML ranking
                                    self.predictions[ind, :5] = valid_rank_thinAR[:5, :] #sizeranks[ind, :]  # change ML predictions and skip spatial reasoning
                            else: # size ranking alone is used
                                #if enough in area + thickness + AR use that
                                if len(candidates_num_thinAR)>=5:
                                    self.predictions[ind,:5] = np.array([(candidates_num_thinAR[x], 0.0) for x in range(5)],dtype='object')
                                #otherwise use area + thickness (in hybrid pipeline it would be fallback to ML but not available here)
                                else:
                                    self.predictions[ind, :5] = np.array([(candidates_num_thin[x], 0.0) for x in range(5)], dtype='object')

                    """Qualitative Spatial Reasoning"""
                    if 'spatial' in self.reasoner_type:
                        QSRs.add_nodes_from(img_ids.keys())
                        #lmapping uses gtruth labels but it is just for visualization purposes
                        #lmapping = dict((o_id, str(i) + '_' + self.remapper[self.labels_full[self.fnames_full.index(o_id)]]) for i,o_id in enumerate(img_ids.keys()))
                        #lmapping['floor'] = 'floor'
                        #lmapping['wall'] = 'wall'
                        for i, o_id in enumerate(img_ids.keys()): # find figures of each reference
                            # tmp_conn, tmp_cur = connect_DB(spatialDB.db_user,
                            #                                                 spatialDB.dbname)  # open spatial DB connection
                            figure_objs = find_neighbours((tmp_conn,tmp_cur), o_id, img_ids)
                            # disconnect_DB(tmp_conn, tmp_cur)  # close spatial DB connection
                            #cobj = lmapping[o_id]
                            #if cobj == '8_backpack':
                                #QSRs = nx.MultiDiGraph()
                            if len(figure_objs)>0: #, if any
                                #Find base QSRs between figure and nearby ref
                                #tmp_conn, tmp_cur = connect_DB(spatialDB.db_user,
                                #                               spatialDB.dbname)  # open spatial DB connection
                                QSRs = extract_QSR((tmp_conn, tmp_cur),o_id,figure_objs,QSRs)
                                #disconnect_DB(tmp_conn, tmp_cur)  # close spatial DB connection
                            # tmp_conn, tmp_cur = connect_DB(spatialDB.db_user,
                            #                                spatialDB.dbname)  # open spatial DB connection
                            if tmp_conn.closed != 0:
                                time.sleep(1)#delay to avoid DB locks
                                #refresh connection, closed by problematic prior query
                                tmp_conn, tmp_cur = connect_DB(spatialDB.db_user, spatialDB.dbname)
                            QSRs = extract_surface_QSR((tmp_conn,tmp_cur),o_id,QSRs) # in any case, alwayes extract relations with walls and floor
                            # disconnect_DB(tmp_conn, tmp_cur)  # close spatial DB connection
                        # after all references in image have been examined
                        # derive special cases of ON
                        QSRs = infer_special_ON(QSRs)
                        #QSRs_H = nx.relabel_nodes(QSRs,lmapping) #human-readable ver
                        #ugr.plot_graph(QSRs_H) #visualize QSR graph for debugging
                        if self.reasoner_type=='size_spatial' and self.withML and not self.waterfall:
                            # 3 judges scenario
                            self.space_validate(tbcorr, QSRs, spatialDB, QSRcandidates, sizerank=thinAR_copy) #pass prior size ranking too
                        elif self.reasoner_type!='size_spatial' and self.withML:
                            self.space_validate(tbcorr, QSRs, spatialDB, QSRcandidates) # proceed with validation/correction based on spatial knowledge
                        elif self.reasoner_type == 'size_spatial' and self.withML and self.waterfall: #size and space alone without ML as base
                            self.space_validate_standalone(tbcorr, QSRs, spatialDB, sizerank=self.predictions)
                        else: #spatial alone without ML ranking as base on correction
                            self.space_validate_standalone(tbcorr, QSRs, spatialDB)
                # else: print("All objects valid wrt size.. skipping correction")
                disconnect_DB(tmp_conn, tmp_cur)

        procTime = float(time.time() - start)  # global proc time
        print("Took % fseconds." % procTime)
        eval_dictionary[self.reasoner_type]['processingTime'].append(procTime)

        #Re-eval post correction
        print("Hybrid results (%s)" % self.reasoner_type)
        eval_dictionary = eval_singlemodel(self, eval_dictionary, self.reasoner_type)
        eval_dictionary = eval_singlemodel(self, eval_dictionary, self.reasoner_type, K=5)
        if 'size' in self.reasoner_type and self.withML:
            #also print results of other combinations of size features
            for abl, preds in list(zip(['size qual', 'size qual+flat', 'size qual+thin', 'size qual+flat+AR', 'size qual+thin+AR']\
                    ,[sizequal_copy, flat_copy, thin_copy, flatAR_copy, thinAR_copy])):
                print("Hybrid results (%s)" % abl)
                self.predictions = preds #ok to change self.predictions directly as in next crossval run the reasoner obj will be refreshed
                eval_dictionary = eval_singlemodel(self, eval_dictionary, abl)
                eval_dictionary = eval_singlemodel(self, eval_dictionary, abl, K=5)

        return eval_dictionary

    def size_select(self, sampled_imgids, all_imgids, threshold_ids, size_KB, session):
        tbcorrected = [] # objects to be corrected in that image
        forQSRs = [] # objects which are validated according to size and will be included in the QSR labels
        for id_ in all_imgids:
            d1,d2,d3 = extract_size(session, id_)
            cropimg_shape = self.crops_full[self.fnames_full.index(id_)]
            #qual, _, aspect_ratio, thinness = quantize([d1,d2,d3], self.lam, self.T,cropimg_shape)
            topMLpred = self.remapper[self.pred_full[self.fnames_full.index(id_), 0, 0]].replace('_',' ')
            # does current ML prediction make sense wrt size KB?
            background = size_KB[topMLpred]
            _,_,_,_,allthree_rank = self.size_validate_ranking([d1,d2,d3],self.lam,self.T,size_KB,cropimg_shape)#self.size_validate_single(qual, thinness, aspect_ratio, background)
            full_vision_rank = self.pred_full[self.fnames_full.index(id_)]
            valid_rank_thinAR = full_vision_rank[[full_vision_rank[z, 0] in allthree_rank for z in range(full_vision_rank.shape[0])]]
            topsizepred = self.remapper[valid_rank_thinAR[0,0]].replace('_',' ')
            # if the size ranking with all 3 features agrees on the top prediction
            # OR machine learning is above conf threshold
            # do not correct and add to QSR label pool
            if topsizepred == topMLpred or id_ in threshold_ids: #all(judgments): # validated based on all 3 size features
                forQSRs.append(id_) # note, all objects count for QSRs, provided there is agreement between ML and size
            else: # otherwise, it will be corrected, if in current subsample list
                if id_ in sampled_imgids:
                    tbcorrected.append(id_)
        return tbcorrected, forQSRs

    def size_validate_single(self, qual, thinness, AR, background):
        return [str(qual) in str(background["has_size"]), str(thinness) in str(background["has_size"]), str(AR) in str(background["aspect_ratio"])]

    def size_validate_ranking(self, estimated_dims, lam, T, KB, crop_shape):

        qual, flat, aspect_ratio, thinness = quantize(estimated_dims, lam, T, crop_shape)
        """ Hybrid (area) """
        candidates = [oname for oname in KB.keys() if qual in str(KB[oname]["has_size"])]  # len([s for s in self.KB[oname]["has_size"] if s.startswith(qual)])>0]
        candidates_num = [self.mapper[oname.replace(' ', '_')] for oname in candidates]

        """ Hybrid (area + flat) """
        candidates_flat = [oname for oname in candidates if str(flat) in str(KB[oname]["is_flat"])]
        candidates_num_flat = [self.mapper[oname.replace(' ', '_')] for oname in candidates_flat]

        """ Hybrid (area + thin) """
        try:
            candidates_thin = [oname for oname in candidates if thinness in str(KB[oname]["thinness"])]
        except KeyError:  # annotation format variation
            candidates_thin = [oname for oname in candidates if thinness in str(KB[oname]["has_size"])]

        candidates_num_thin = [self.mapper[oname.replace(' ', '_')] for oname in candidates_thin]

        """ Hybrid (area + flat+AR) """
        candidates_flat_AR = [oname for oname in candidates_flat if aspect_ratio in str(KB[oname]["aspect_ratio"])]
        candidates_num_flatAR = [self.mapper[oname.replace(' ', '_')] for oname in candidates_flat_AR]

        """ Hybrid (area + thin +AR) """
        candidates_thin_AR = [oname for oname in candidates_thin if aspect_ratio in str(KB[oname]["aspect_ratio"])]
        candidates_num_thinAR = [self.mapper[oname.replace(' ', '_')] for oname in candidates_thin_AR]

        return [candidates_num, candidates_num_flat, candidates_num_thin, candidates_num_flatAR, candidates_num_thinAR]

    def space_validate(self,obj_list,qsr_graph,spatialDB, sizevalidated_ids, K=5, sizerank=None):

        for oid in obj_list: #for each object to correct/validate
            i = self.fnames.index(oid)
            ML_rank = self.predictions[i, :K]
            spatialforML = np.copy(ML_rank)
            if sizerank is not None:  #both size and ML are leveraged, spatial acts as 3rd judgement
                # 3 judges case
                size_rank = sizerank[i,:K]
                spatialforsize = np.copy(size_rank)
                if self.remapper[size_rank[0][0]] =='person': #skip space validation whenever top ML prediction is person
                #people cannot be discriminated based on where they lie
                    self.predictions[i, :K] = size_rank
                    print("top size prediction is person, skipping spatial validation")
                    continue
            #hybrid_rank = np.copy(ML_rank)
            print("%s predicted as %s" % (self.remapper[self.labels[i]],self.remapper[ML_rank[0][0]]))

            print("Top-5 before spatial validation: ")
            read_current_rank = [(self.remapper[ML_rank[z, 0]], ML_rank[z, 1]) for z in
                                 range(ML_rank.shape[0])]
            print(read_current_rank) #ML rank in human readable form

            #option to reduce skew/bias towards all classes but person after spatial reasoning
            if read_current_rank[0][0] =='person':
                print("top ML prediction is person, skipping spatial validation")
                continue #skip space validation whenever top ML prediction is person
                #people cannot be discriminated based on where they lie

            for n, (cnum, L2dis) in enumerate(ML_rank): #for each class in the ML rank
                pred_label = self.remapper[cnum]
                wn_syn = self.taxonomy[pred_label] #wordnet synset for that label

                if sizerank is not None:
                    sclass, _ = size_rank[n]
                    size_label = self.remapper[sclass]
                    wn_syn_size = self.taxonomy[size_label]

                if self.spatial_label_type == 'gold':
                    # use ground truth for nearby object (except the one being predicted)
                    fig_qsrs = [(pred_label,self.remapper[self.labels_full[self.fnames_full.index(ref)]],r['QSR'])
                            for f,ref,r in qsr_graph.out_edges(oid, data=True) if ref not in ['wall','floor']] #rels where obj is figure
                    ref_qsrs = [(self.remapper[self.labels_full[self.fnames_full.index(f)]],pred_label,r['QSR'])
                            for f,ref,r in qsr_graph.in_edges(oid, data=True) if f not in ['wall','floor']] # rels where obj is reference

                elif self.spatial_label_type == 'ML':
                    # independent from order because we collect it from the separate list self.pred_full and only modify self.predictions
                    # at the end of validation
                    # use ML predictions that are above conf threshold
                    fig_qsrs = [(pred_label, self.remapper[self.pred_full[self.fnames_full.index(ref), 0, 0]], r['QSR'])
                                for f, ref, r in qsr_graph.out_edges(oid, data=True) if
                                ref not in ['wall', 'floor']
                                and self.pred_full[self.fnames_full.index(ref), 0, 1] < self.epsilon_set[0]]  # rels where obj is figure
                    ref_qsrs = [(self.remapper[self.pred_full[self.fnames_full.index(f), 0, 0]], pred_label, r['QSR'])
                                for f, ref, r in qsr_graph.in_edges(oid, data=True) if
                                f not in ['wall', 'floor']
                                and self.pred_full[self.fnames_full.index(f), 0, 1] < self.epsilon_set[0]]  # rels where obj is reference

                elif self.spatial_label_type == 'sizevalidated':
                    fig_qsrs = [(pred_label, self.remapper[self.pred_full[self.fnames_full.index(ref), 0, 0]], r['QSR'])
                                for f, ref, r in qsr_graph.out_edges(oid, data=True) if
                                ref in sizevalidated_ids]
                    ref_qsrs = [(self.remapper[self.pred_full[self.fnames_full.index(f), 0, 0]], pred_label, r['QSR'])
                                for f, ref, r in qsr_graph.in_edges(oid, data=True) if
                                f in sizevalidated_ids]

                elif self.spatial_label_type == 'hybrid':
                    # option to consider already corrected predictions, if available
                    # discard all below conf threshold though

                    fig_qsrs = [(pred_label, ref, r['QSR']) for _, ref, r in qsr_graph.out_edges(oid, data=True)
                                if ref not in ['wall', 'floor'] and
                                self.pred_full[self.fnames_full.index(ref), 0, 1] < self.epsilon_set[0]]  # rels where obj is figure
                    fig_qsrs = [(f,self.remapper[self.pred_full[self.fnames_full.index(ref), 0, 0]],r)
                                if ref not in self.fnames else (f, self.remapper[self.predictions[self.fnames.index(ref), 0, 0]], r)
                                for f,ref,r in fig_qsrs]
                    ref_qsrs = [(f, pred_label, r['QSR']) for f, _, r in qsr_graph.in_edges(oid, data=True) if f not in ['wall', 'floor']\
                                and self.pred_full[self.fnames_full.index(f), 0, 1] < self.epsilon_set[0]]  # rels where obj is reference
                    ref_qsrs = [(self.remapper[self.pred_full[self.fnames_full.index(f), 0, 0]], ref, r)
                                if f not in self.fnames else (self.remapper[self.predictions[self.fnames.index(f),0,0]], ref, r)
                                for f,ref,r in ref_qsrs]

                #Retrieve wall and floor QSRs, only in figure/reference form - e.g., 'object onTopOf
                surface_qsrs = [(pred_label,ref,r['QSR']) for f,ref,r \
                                in qsr_graph.out_edges(oid, data=True) if ref in ['wall','floor']] #only those in fig/ref form
                fig_qsrs.extend(surface_qsrs) # merge into list of fig/ref relations

                """print("Figure-Ref QSRS are")
                print(fig_qsrs)
                print("Ref-Figure QSRS are")
                print(ref_qsrs)"""

                if sizerank is not None and \
                        (not wn_syn_size or size_label=='person' or not wn_syn or pred_label =='person' or (len(ref_qsrs)==0 and len(fig_qsrs)==0)):
                    # 3 judges case
                    spatialforsize[n][1] += 1.
                    spatialforML[n][1] += 1.
                    continue
                elif sizerank is None and \
                        (not wn_syn or pred_label =='person' or (len(ref_qsrs)==0 and len(fig_qsrs)==0)): #objects that do not have a mapping to VG through WN (foosball table and pigeon holes)
                    # we skip people as they are mobile and can be anywhere, space is not discriminating
                    # OR there are no QSRs to consider
                    spatialforML[n][1] += 1. #added up to ML #add up 1. as if not found to not alter prior ranking and skip
                    continue

                #Tipicality scores based on VG stats
                sub_syn = wn_syn
                all_spatial_scores = []
                if sizerank is not None: # 3 judges case
                    sub_syn_size = wn_syn_size
                    all_spatial_scores_size = []
                fig_rs = list(set([r for _,_,r in fig_qsrs])) #distinct figure relations present
                for _,ref,r in fig_qsrs: #for each QSR where obj is figure, i.e., subject
                    if ref=='wall': obj_syn = ['wall.n.01'] #cases where reference is wall or floor
                    elif ref=='floor': obj_syn = ['floor.n.01']
                    else: obj_syn = self.taxonomy[ref]
                    if obj_syn =='': #reference obj is e.g., foosball table or pigeon holes (absent from background KB)
                        all_spatial_scores.append(1.) #add up 1. as if not found to not alter ML ranking and skip
                        if sizerank is not None: # 3 judges case
                            all_spatial_scores_size.append(1.)
                        continue
                    if r == 'touches':
                        if len(fig_rs) ==1: # touches is the only rel
                            all_spatial_scores.append(1.)  # add up 1. as if not found to not alter ML ranking and skip
                            if sizerank is not None: # 3 judges case
                                all_spatial_scores_size.append(1.)
                            continue
                        else: continue #there are other types, just skip this one as not relevant for VG
                    elif r=='beside':
                        continue #beside already checked through L/R rel
                    elif r == 'leansOn' or r == 'affixedOn': r = 'against'  # mapping on VG predicate
                    all_spatial_scores = self.compute_all_scores(spatialDB, all_spatial_scores,sub_syn, obj_syn,r)
                    if sizerank is not None: # 3 judges case
                        all_spatial_scores_size = self.compute_all_scores(spatialDB, all_spatial_scores_size,sub_syn_size, obj_syn,r)

                # Similarly, for QSRs where predicted obj is reference, i.e., object
                obj_syn = wn_syn #self.taxonomy[pred_label]
                if sizerank is not None: obj_syn_size = wn_syn_size#self.taxonomy[size_label]
                ref_rs = list(set([r for _, _, r in ref_qsrs]))  # distinct figure relations present
                for fig,_,r in ref_qsrs:
                    if fig=='wall': sub_syn = ['wall.n.01'] #cases where reference is wall or floor
                    elif fig=='floor': sub_syn = ['floor.n.01']
                    else: sub_syn = self.taxonomy[fig]
                    if sub_syn =='': #figure obj is e.g., foosball table or pigeon holes (absent from background KB)
                        all_spatial_scores.append(1.) #add up 1. as if not found to not alter ML ranking and skip
                        if sizerank is not None: # 3 judges case
                            all_spatial_scores_size.append(1.)
                        continue
                    if r == 'touches':
                        if len(ref_rs) ==1: # touches is the only rel
                            all_spatial_scores.append(1.)  # add up 1. as if not found to not alter ML ranking and skip
                            if sizerank is not None: # 3 judges case
                                all_spatial_scores_size.append(1.)
                            continue
                        else: continue #there are other types, just skip this one as not relevant for VG
                    elif r=='beside': continue  # touches not useful for VG predicates, beside already checked through L/R rel
                    elif r =='leansOn' or r=='affixedOn': r = 'against' #mapping on VG predicate
                    all_spatial_scores = self.compute_all_scores(spatialDB, all_spatial_scores, sub_syn, obj_syn, r)
                    if sizerank is not None: # 3 judges case
                        all_spatial_scores_size = self.compute_all_scores(spatialDB, all_spatial_scores_size, sub_syn, obj_syn_size, r)

                # Average across all QSRs
                avg_spatial_score = statistics.mean(all_spatial_scores)
                spatialforML[n][1] += avg_spatial_score  # add up to ML score
                if sizerank is not None:
                    # adding up to original score
                    avg_spatial_score_size = statistics.mean(all_spatial_scores_size)

                    # adding up to original score
                    spatialforsize[n][1] += avg_spatial_score_size

            # Normalise scores across classes, so it is between 0 and 1
            # minmax norm
            scores = spatialforML[:,1]
            min_, max_ = np.min(scores), np.max(scores)
            if max_ - min_ !=0: #avoid division by zero
                spatialforML[:, 1] = np.array([(x-min_)/(max_ - min_) for x in scores])
            posthoc_rank = spatialforML[np.argsort(spatialforML[:, 1])] # order by score ascending
            # ranking after correction is ..
            print("Top-5 after spatial validation on ML rank: ")
            read_phoc_rank = [(self.remapper[posthoc_rank[z, 0]], posthoc_rank[z, 1]) for z in range(posthoc_rank.shape[0])]
            print(read_phoc_rank)  # posthoc rank in human readable form

            #add plausible spatial objects to joint voting
            if sizerank is not None: # 3 judges case
                spatial_classes = [cnum for cnum,jdis in posthoc_rank if jdis < 1.]
                scores = spatialforsize[:, 1]
                # apply minmax norm to spatialforsize rank too
                min_, max_ = np.min(scores), np.max(scores)
                if max_ - min_ != 0:  # avoid division by zero
                    spatialforsize[:, 1] = np.array([(x - min_) / (max_ - min_) for x in scores])
                posthoc_rank_size = spatialforsize[np.argsort(spatialforsize[:, 1])]  # order by score ascending
                print("Top-5 after spatial validation on size rank: ")
                read_phoc_rank_size = [(self.remapper[posthoc_rank_size[z, 0]], posthoc_rank_size[z, 1]) for z in
                                  range(posthoc_rank_size.shape[0])]
                print(read_phoc_rank_size)  # posthoc rank in human readable form
                spatial_classes.extend([cnum for cnum, jdis in posthoc_rank_size if jdis < 1.]) # full class set
                if len(spatial_classes)==0:
                    #there were no useful scores through spatial reasoning
                    self.predictions[i, :K] = size_rank
                    print("Keep size ranking")
                    continue

                votes = {cl: 0. for cl in spatial_classes} #votes = {cl: 0. for cl in spatial_classes}
                # votes = Counter(spatial_classes)
                # Borda count positional voting system
                # the higher the position in the ranking the higher the points assigned

                # spatialML_ord = [(cnum, K-(m+1)) for m,(cnum,jdis) in enumerate(posthoc_rank) if jdis < 1.]
                for m,(cnum,jdis) in enumerate(posthoc_rank):
                    if jdis < 1.:
                        votes[cnum] += K-(m+1)  #.append(jdis*(m+1))

                # spatialsize_ord = [(cnum, K-(m+1)) for m,(cnum,jdis) in enumerate(posthoc_rank_size) if jdis < 1.]

                for m, (cnum, jdis) in enumerate(posthoc_rank_size):
                    if jdis < 1.:
                        votes[cnum]+= K - (m + 1) #.append(jdis*(m+1)) #+= K - (m + 1)

                # average class-wise
                # finrank_list = [(l, statistics.mean(val)) for l,val in votes.items()]
                # finrank_list.sort(key=lambda x: x[1]) # sort by score desc
                #if there are ties
                # if finrank_list[0][1] == finrank_list[1][1]:
                #     self.predictions[i, :K] = size_rank
                #     print("Keep size validated ranking")
                #     continue
                s_ = list(votes.values())
                # c_ = list(votes.keys())
                if s_[0] == s_[1]:
                    self.predictions[i, :K] = size_rank
                    print("Keep size validated ranking")
                    continue

                else:
                    # Fill up list with duplicates in order from most common to least common
                    # so that final ranking is still K positions long
                    finrank_list = [(l, num) for l, num in votes.items()]
                    if len(finrank_list)< K:
                        to_fill = K - len(finrank_list)
                        topl, tops = finrank_list[0][0], finrank_list[0][1]

                        for num in range(to_fill): # fill remaining positions with top scores
                            finrank_list.append((topl,tops))
                        finrank_list.sort(key=lambda x: x[1], reverse=True) #reorder list in the end, scores descending
                        """for k, (l, num) in enumerate(votes.keys()): #enumerate(votes.most_common(K)):
                            
                            remaining_spots = K - len(finrank_list)
                            if num<= remaining_spots: cap = num
                            else: cap = remaining_spots
                            for r in range(cap): finrank_list.append((l,0.))
                            remaining_spots = K - len(finrank_list)
                            if remaining_spots==0: #reached end of topK
                                break #stop at topK"""
                    else:
                        finrank_list = finrank_list[:K] #only keep top-5
                    final_rank = np.array(finrank_list,dtype='object') #order by number of votes

            else:
                # use spatial ranking combined with Ml score directly
                final_rank = posthoc_rank

            self.predictions[i, :K] = final_rank
            print("Final top-5: ")
            read_final_rank = [self.remapper[final_rank[z, 0]] for z in range(final_rank.shape[0])]
            print(read_final_rank)
            continue

    def space_validate_standalone(self,obj_list,qsr_graph,spatialDB, K=5, sizerank=None):

        all_classes = list(self.mapper.keys())

        for oid in obj_list: #for each object to correct/validate
            i = self.fnames.index(oid)
            ML_rank = self.predictions[i, :K]
            spatialonly = np.empty((len(all_classes), 2), dtype='object')
            spatialonly[:, 0] = np.array(list(self.mapper.values()))
            spatialonly[:, 1] = np.ones((len(all_classes),))

            if sizerank is not None:
                size_rank = sizerank[i,:K] # either ML rank or size-validate ML rank (if waterfall option selected)
                spatialforsize = np.copy(size_rank)
                if self.remapper[size_rank[0][0]] =='person': #skip space validation whenever top ML prediction is person
                #people cannot be discriminated based on where they lie
                    self.predictions[i, :K] = size_rank
                    print("top size prediction is person, skipping spatial validation")
                    continue

                print("%s predicted as %s" % (self.remapper[self.labels[i]],self.remapper[size_rank[0][0]]))

                print("Top-5 before spatial validation: ")
                read_current_rank = [(self.remapper[size_rank[z, 0]], size_rank[z, 1]) for z in
                                     range(size_rank.shape[0])]
                print(read_current_rank) #ML rank in human readable form

                for n, (cnum, _) in enumerate(size_rank): #for each class in the prior rank

                    size_label = self.remapper[cnum]
                    wn_syn_size = self.taxonomy[size_label]

                    if self.spatial_label_type == 'gold':
                        # use ground truth for nearby object (except the one being predicted)
                        fig_qsrs = [(size_label, self.remapper[self.labels_full[self.fnames_full.index(ref)]], r['QSR'])
                                    for f, ref, r in qsr_graph.out_edges(oid, data=True) if
                                    ref not in ['wall', 'floor']]  # rels where obj is figure
                        ref_qsrs = [(self.remapper[self.labels_full[self.fnames_full.index(f)]], size_label, r['QSR'])
                                    for f, ref, r in qsr_graph.in_edges(oid, data=True) if
                                    f not in ['wall', 'floor']]  # rels where obj is reference

                    elif self.spatial_label_type == 'ML':
                        # independent from order because we collect it from the separate list self.pred_full and only modify self.predictions
                        # at the end of validation
                        # use ML predictions that are above conf threshold
                        fig_qsrs = [(size_label, self.remapper[self.pred_full[self.fnames_full.index(ref), 0, 0]], r['QSR'])
                            for f, ref, r in qsr_graph.out_edges(oid, data=True) if
                            ref not in ['wall', 'floor']
                            and self.pred_full[self.fnames_full.index(ref), 0, 1] < self.epsilon_set[
                                0]]  # rels where obj is figure
                        ref_qsrs = [
                            (self.remapper[self.pred_full[self.fnames_full.index(f), 0, 0]], size_label, r['QSR'])
                            for f, ref, r in qsr_graph.in_edges(oid, data=True) if
                            f not in ['wall', 'floor']
                            and self.pred_full[self.fnames_full.index(f), 0, 1] < self.epsilon_set[
                                0]]  # rels where obj is reference

                    #Retrieve wall and floor QSRs, only in figure/reference form - e.g., 'object onTopOf
                    surface_qsrs = [(size_label,ref,r['QSR']) for f,ref,r \
                                    in qsr_graph.out_edges(oid, data=True) if ref in ['wall','floor']] #only those in fig/ref form
                    fig_qsrs.extend(surface_qsrs) # merge into list of fig/ref relations

                    if not wn_syn_size or size_label=='person' or (len(ref_qsrs)==0 and len(fig_qsrs)==0):
                        spatialforsize[n][1] += 1. #added up to original ML score
                        continue

                    sub_syn_size = wn_syn_size
                    all_spatial_scores_size = []
                    fig_rs = list(set([r for _,_,r in fig_qsrs])) #distinct figure relations present
                    for _,ref,r in fig_qsrs: #for each QSR where obj is figure, i.e., subject
                        if ref=='wall': obj_syn = ['wall.n.01'] #cases where reference is wall or floor
                        elif ref=='floor': obj_syn = ['floor.n.01']
                        else: obj_syn = self.taxonomy[ref]
                        if obj_syn =='': #reference obj is e.g., foosball table or pigeon holes (absent from background KB)
                            all_spatial_scores_size.append(1.)
                            continue
                        if r == 'touches':
                            if len(fig_rs) ==1: # touches is the only rel
                                all_spatial_scores_size.append(1.)
                                continue
                            else: continue #there are other types, just skip this one as not relevant for VG
                        elif r=='beside':
                            continue #beside already checked through L/R rel
                        elif r == 'leansOn' or r == 'affixedOn': r = 'against'  # mapping on VG predicate
                        all_spatial_scores_size = self.compute_all_scores(spatialDB, all_spatial_scores_size,sub_syn_size, obj_syn,r)

                    # Similarly, for QSRs where predicted obj is reference, i.e., object
                    obj_syn_size = wn_syn_size#self.taxonomy[size_label]
                    ref_rs = list(set([r for _, _, r in ref_qsrs]))  # distinct figure relations present
                    for fig,_,r in ref_qsrs:
                        if fig=='wall': sub_syn = ['wall.n.01'] #cases where reference is wall or floor
                        elif fig=='floor': sub_syn = ['floor.n.01']
                        else: sub_syn = self.taxonomy[fig]
                        if sub_syn =='': #figure obj is e.g., foosball table or pigeon holes (absent from background KB)
                            all_spatial_scores_size.append(1.)
                            continue
                        if r == 'touches':
                            if len(ref_rs) ==1: # touches is the only rel
                                all_spatial_scores_size.append(1.)
                                continue
                            else: continue #there are other types, just skip this one as not relevant for VG
                        elif r=='beside': continue  # touches not useful for VG predicates, beside already checked through L/R rel
                        elif r =='leansOn' or r=='affixedOn': r = 'against' #mapping on VG predicate
                        all_spatial_scores_size = self.compute_all_scores(spatialDB, all_spatial_scores_size, sub_syn, obj_syn_size, r)

                    # Average across all QSRs
                    # added up to original ML score
                    avg_spatial_score_size = statistics.mean(all_spatial_scores_size)
                    spatialforsize[n][1] += avg_spatial_score_size

                # Normalise scores across classes, so it is between 0 and 1
                # minmax norm
                scores = spatialforsize[:,1]
                min_, max_ = np.min(scores), np.max(scores)
                if max_ - min_ !=0: #avoid division by zero
                    spatialforsize[:, 1] = np.array([(x-min_)/(max_ - min_) for x in scores])
                posthoc_rank = spatialforsize[np.argsort(spatialforsize[:, 1])] # order by score ascending

            else:
                print("%s predicted as %s" % (self.remapper[self.labels[i]], self.remapper[ML_rank[0][0]]))
                for y, c in enumerate(all_classes): #no background ranking, run for all taxonomy classes and maximise scores at the end
                    wn_syn = self.taxonomy[c]
                    # changed, if only spatial no option but to use gold labels
                    # use ground truth for nearby object (except the one being predicted)
                    fig_qsrs = [(c, self.remapper[self.labels_full[self.fnames_full.index(ref)]], r['QSR'])
                                for f, ref, r in qsr_graph.out_edges(oid, data=True) if
                                ref not in ['wall', 'floor']]  # rels where obj is figure
                    ref_qsrs = [(self.remapper[self.labels_full[self.fnames_full.index(f)]], c, r['QSR'])
                                for f, ref, r in qsr_graph.in_edges(oid, data=True) if
                                f not in ['wall', 'floor']]  # rels where obj is reference

                    surface_qsrs = [(c, ref, r['QSR']) for f, ref, r \
                                    in qsr_graph.out_edges(oid, data=True) if
                                    ref in ['wall', 'floor']]  # only those in fig/ref form
                    fig_qsrs.extend(surface_qsrs)  # merge into list of fig/ref relations

                    if not wn_syn or (len(ref_qsrs) == 0 and len(fig_qsrs) == 0):
                        continue
                    sub_syn = wn_syn
                    all_spatial_scores = []
                    fig_rs = list(set([r for _, _, r in fig_qsrs]))  # distinct figure relations present
                    for _, ref, r in fig_qsrs:  # for each QSR where obj is figure, i.e., subject
                        if ref == 'wall':
                            obj_syn = ['wall.n.01']  # cases where reference is wall or floor
                        elif ref == 'floor':
                            obj_syn = ['floor.n.01']
                        else:
                            obj_syn = self.taxonomy[ref]
                        if obj_syn == '':  # reference obj is e.g., foosball table or pigeon holes (absent from background KB)
                            all_spatial_scores.append(1.)
                            continue
                        if r == 'touches':
                            if len(fig_rs) == 1:  # touches is the only rel
                                all_spatial_scores.append(1.)
                                continue
                            else:
                                continue  # there are other types, just skip this one as not relevant for VG
                        elif r == 'beside':
                            continue  # beside already checked through L/R rel
                        elif r == 'leansOn' or r == 'affixedOn':
                            r = 'against'  # mapping on VG predicate
                        all_spatial_scores = self.compute_all_scores(spatialDB, all_spatial_scores,sub_syn, obj_syn, r)

                    # Similarly, for QSRs where predicted obj is reference, i.e., object
                    obj_syn = wn_syn  # self.taxonomy[size_label]
                    ref_rs = list(set([r for _, _, r in ref_qsrs]))  # distinct figure relations present
                    for fig, _, r in ref_qsrs:
                        if fig == 'wall':
                            sub_syn = ['wall.n.01']  # cases where reference is wall or floor
                        elif fig == 'floor':
                            sub_syn = ['floor.n.01']
                        else:
                            sub_syn = self.taxonomy[fig]
                        if sub_syn == '':  # figure obj is e.g., foosball table or pigeon holes (absent from background KB)
                            all_spatial_scores.append(1.)
                            continue
                        if r == 'touches':
                            if len(ref_rs) == 1:  # touches is the only rel
                                all_spatial_scores.append(1.)
                                continue
                            else:
                                continue  # there are other types, just skip this one as not relevant for VG
                        elif r == 'beside':
                            continue  # touches not useful for VG predicates, beside already checked through L/R rel
                        elif r == 'leansOn' or r == 'affixedOn':
                            r = 'against'  # mapping on VG predicate
                        all_spatial_scores = self.compute_all_scores(spatialDB, all_spatial_scores, sub_syn,
                                                                          obj_syn, r)
                    # Average across all QSRs
                    # changed: rewrite original score instead of adding up
                    avg_spatial_score = statistics.mean(all_spatial_scores)
                    spatialonly[y][1] = avg_spatial_score

                    # Normalise scores across classes, so it is between 0 and 1
                    # minmax norm
                scores = spatialonly[:, 1]
                min_, max_ = np.min(scores), np.max(scores)
                if max_ - min_ != 0:  # avoid division by zero
                    spatialonly[:, 1] = np.array([(x - min_) / (max_ - min_) for x in scores])
                posthoc_rank = spatialonly[np.argsort(spatialonly[:, 1])][:K,:] # order by score ascending

            # ranking after correction is ..
            print("Top-5 after spatial validation on ML rank: ")
            read_phoc_rank = [(self.remapper[posthoc_rank[z, 0]], posthoc_rank[z, 1]) for z in
                              range(posthoc_rank.shape[0])]
            print(read_phoc_rank)  # posthoc rank in human readable form
            final_rank = posthoc_rank
            self.predictions[i, :K] = final_rank

    def compute_all_scores(self, spatialDB, all_scores, sub_syn, obj_syn, r):

        if len(sub_syn) > 1 or len(obj_syn) > 1:
            if len(sub_syn) > 1 and len(obj_syn) > 1:  # try all sub,obj ordered combos
                # print(list(itertools.product(sub_syn, obj_syn)))
                typscores = [self.compute_typicality_score(spatialDB, sub_s, obj_s, r) \
                             for sub_s, obj_s in list(itertools.product(sub_syn, obj_syn))]
            elif len(sub_syn) == 1 and len(obj_syn) > 1:
                sub_syn = sub_syn[0]
                typscores = [self.compute_typicality_score(spatialDB, sub_syn, osyn, r) for osyn in obj_syn]
            elif len(sub_syn) > 1 and len(obj_syn) == 1:
                obj_syn = obj_syn[0]
                typscores = [self.compute_typicality_score(spatialDB, subs, obj_syn, r) for subs in sub_syn]

            typscores = [s for s in typscores if s != 0.]  # keep only synset that of no-null typicality
            # in order of taxonomy (from preferred synset to least preferred)
            if len(typscores) == 0:
                typscore = 0.
            else:
                typscore = typscores[0]  # first one in the order

        else:
            sub_syn, obj_syn = sub_syn[0], obj_syn[0]
            typscore = self.compute_typicality_score(spatialDB, sub_syn, obj_syn, r)
        all_scores.append((1. - typscore))  # track INVERSE of score (so that it is comparable
        # with L2 distances, i.e., scores that are minimised)
        #print("Jaccard distance: %f" % (1. - typscore))
        return all_scores

    def compute_typicality_score(self,spatialDB,sub_syn,obj_syn,rel, use_beside=False, use_near=False):
        #no of times the two appeared in that relation in VG
        try:
            nom = float(spatialDB.KB.VG_stats['predicates'][rel]['relations'][str((str(sub_syn), str(obj_syn)))])
        except KeyError: #if any hit is found
            if rel !='above':
                if rel =='leftOn' or rel=='rightOf': #check also hits for (more generic) beside predicate
                    try:
                        nom = float(spatialDB.KB.VG_stats['predicates']['beside']['relations'][str((str(sub_syn), str(obj_syn)))])
                        use_beside = True
                    except KeyError:
                        return 0.
                else: #check also hits for (more generic) near predicate
                    try:
                        nom = float(spatialDB.KB.VG_stats['predicates']['near']['relations'][str((str(sub_syn), str(obj_syn)))])
                        use_near = True
                    except KeyError:
                        return 0.
            else: return 0.
        #no of times sub_syn was subject of r relation in VG
        try:
            if use_near:
                denom1 = float(spatialDB.KB.VG_stats['subjects'][sub_syn]['near'])
            elif use_beside:
                denom1 = float(spatialDB.KB.VG_stats['subjects'][sub_syn]['beside'])
            else: denom1 = float(spatialDB.KB.VG_stats['subjects'][sub_syn][rel])
        except KeyError: #if any hit is found
            return 0.
        #no of times obj_syn was object of r relation in VG
        try:
            if use_near:
                denom2 = float(spatialDB.KB.VG_stats['objects'][obj_syn]['near'])
            elif use_beside:
                denom2 = float(spatialDB.KB.VG_stats['objects'][obj_syn]['beside'])
            else: denom2 = float(spatialDB.KB.VG_stats['objects'][obj_syn][rel])
        except KeyError: #if any hit is found
            return 0.
        return nom / (denom1+denom2-nom)

    def makereadable(self,ML_rank, valid_rank, valid_rank_flat, valid_rank_thin, valid_rank_flatAR, valid_rank_thinAR):

        read_rank_ML = [(self.remapper[ML_rank[z, 0]], ML_rank[z, 1]) for z in
                     range(ML_rank.shape[0])]
        read_rank_area = [(self.remapper[valid_rank[z, 0]], valid_rank[z, 1]) for z in
                     range(valid_rank.shape[0])]
        read_rank_flat = [(self.remapper[valid_rank_flat[z, 0]], valid_rank_flat[z, 1]) for z in
                          range(valid_rank_flat.shape[0])]
        read_rank_thin = [(self.remapper[valid_rank_thin[z, 0]], valid_rank_thin[z, 1]) for z in
                          range(valid_rank_thin.shape[0])]
        if len(valid_rank_flatAR)>0:
            read_rank_flatAR = [(self.remapper[valid_rank_flatAR[z, 0]], valid_rank_flatAR[z, 1]) for z in
                                range(valid_rank_flatAR.shape[0])]
        else: read_rank_flatAR =[]
        if len(valid_rank_thinAR)>0:
            read_rank_thinAR = [(self.remapper[valid_rank_thinAR[z, 0]], valid_rank_thinAR[z, 1]) for z in
                            range(valid_rank_thinAR.shape[0])]
        else: read_rank_thinAR=[]

        return [read_rank_ML, read_rank_area, read_rank_flat, read_rank_thin, read_rank_flatAR, read_rank_thinAR]