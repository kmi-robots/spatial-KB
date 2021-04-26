"""KB class"""
import os
import time
import requests

#from onto import init_onto
from PostGIS import *
from VG import *


class KnowledgeBase():
    def __init__(self, args):
        # Load raw external KB data
        self.path_to_VGrel = os.path.join(args.path_to_data, 'relationships.json')
        self.path_to_VGstats = os.path.join(args.path_to_data, 'VG_spatial_stats.json')
        self.path_to_predicate_aliases = os.path.join(args.path_to_data, 'relationship_aliases.txt')
        self.predicate_set = ["in", "on", "in front of", "behind", "to left of", "to right of",
                 "next to", "under", "above"] #Derived from Spatial Sense (Yang et al., 2019)

    def load_data(self,reasoner):
        # Same db session as reasoner
        self.cursor, self.connection = reasoner.cursor, reasoner.connection

        # Initialise table structure, if not exists
        create_VG_table(self.cursor)

        # Check if data were already pre-processed before
        if not os.path.isfile(self.path_to_VGstats): #data preparation needed
            self.VG_stats = self.data_prep() # also insert data in DB table
        else:
            with open(self.path_to_VGstats) as fin:
                self.VG_stats = json.load(fin)
        return self

    def data_prep(self):
        print("Preparing spatial data first...")
        start = time.time()
        # Load VG raw relationships and aliases
        raw_data, alias_index = load_rel_bank(self)
        # flatten alias index
        alias_mergedlist = list(alias_index.values())

        VG_stats = {k: {} for k in ["predicates", "subjects", "objects"]}
        for entry in raw_data:
            img_rels = entry["relationships"]  # all relationships for a given image
            for rel in img_rels:
                pred = rel['predicate'].lower()
                # for predicates we need aliases (if any is found) because synsets can be unknown or too generic
                aliases = []
                for alias_set in alias_mergedlist:
                    if pred in alias_set:
                        aliases = alias_set
                        break
                #intersection = list(set(aliases) & set(self.predicate_set))
                sub_syn = rel['subject']['synsets']
                obj_syn = rel['object']['synsets']

                if len(sub_syn) != 1 or len(obj_syn) != 1 or pred == '' or pred == ' ':
                    # Skipping:
                    # (ii) relations without both sub and object synsets
                    # (iii) compound periods, i.e., more than one synset per entity
                    # e.g.  subject: green trees seen  pred: green trees by road object: trees on roadside.)
                    # e.g.  subject: see cupboard  pred: cupboard black object: cupboard not white. )
                    # (iv) as well as empty predicates
                    continue

                #Below/Under
                if pred in alias_index["under"]:
                    # update VG predicate statistics
                    VG_stats = update_VG_stats(VG_stats, "belowOf", aliases, alias_index, sub_syn, obj_syn)

                if "above" in pred:
                    VG_stats = update_VG_stats(VG_stats, "aboveOf", aliases, alias_index, sub_syn, obj_syn)

                elif "right of" in pred:
                    VG_stats = update_VG_stats(VG_stats, "rightOf", aliases, alias_index, sub_syn, obj_syn)
                    # union of right and left also counts as beside
                    VG_stats = update_VG_stats(VG_stats, "beside", aliases, alias_index, sub_syn, obj_syn)

                elif "left of" in pred:
                    VG_stats = update_VG_stats(VG_stats, "leftOf", aliases, alias_index, sub_syn, obj_syn)
                    # union of right and left also counts as beside
                    VG_stats = update_VG_stats(VG_stats, "beside", aliases, alias_index, sub_syn, obj_syn)

                elif "in front" in pred:
                    VG_stats = update_VG_stats(VG_stats, "inFrontOf", aliases, alias_index, sub_syn, obj_syn)

                elif pred in alias_index["behind"]:
                    VG_stats = update_VG_stats(VG_stats, "behindOf", aliases, alias_index, sub_syn, obj_syn)

                elif "near" in pred or 'at' in pred:
                    VG_stats = update_VG_stats(VG_stats, "near", aliases, alias_index, sub_syn, obj_syn)

                elif "beside" in pred or 'next to' in pred or 'by' in pred:
                    VG_stats = update_VG_stats(VG_stats, "beside", aliases, alias_index, sub_syn, obj_syn)

                elif "on top of" in pred:
                    VG_stats = update_VG_stats(VG_stats, "onTopOf", aliases, alias_index, sub_syn, obj_syn)

                elif 'inside' in pred:
                    VG_stats = update_VG_stats(VG_stats, "insideOf", aliases, alias_index, sub_syn, obj_syn)

                if 'on' in pred: #TODO handle cases and consider that on may have already appeared above too
                    if 'top' in pred:
                        VG_stats = update_VG_stats(VG_stats, "onTopOf", aliases, alias_index, sub_syn, obj_syn)

                if 'in' in pred: #TODO handle cases and consider that on may have already appeared above too
                    continue

                # Disambiguate ON uses based on postgis operators on 2D bboxes
                if pred in alias_index["on"]:
                    # 1. Populate spatial DB #
                    rel_id = rel['relationship_id'] #to use as primary key

                    # 2D Bounding box corners of subject (e.g., "cup ON table" subj = cup, obj = table)
                    # Format: from top-left corner anti-clockwise
                    # PostGIS requires to repeat top-left twice to close the ring
                    x1, y1 = rel['subject']['x'], rel['subject']['y']
                    x2, y2 = x1, (y1 + rel['subject']['h'])
                    x3, y3 = (x1 + rel['subject']['w']), (y1 + rel['subject']['h'])
                    x4, y4 = (x1 + rel['subject']['w']), y1
                    sub_coords = ((x1, y1), (x2, y2), (x3, y3), (x4, y4), (x1, y1))

                    # top 2D half-space projection of object bbox
                    x1, y1 = rel['object']['x'], (rel['object']['y'] - rel['object']['h'])
                    x2, y2 = rel['object']['x'], rel['object']['y']
                    x3, y3 = (x1 + rel['object']['w']), rel['object']['y']
                    x4, y4 = (x1 + rel['object']['w']), (rel['object']['y']-rel['object']['h'])
                    obj_top_proj_coords = ((x1, y1), (x2, y2), (x3, y3), (x4, y4), (x1, y1))

                    add_VG_row(self.cursor,[rel_id,pred,aliases, sub_coords, obj_top_proj_coords])
                    self.connection.commit() # new row needs to be visible/up-to-date to compute operations on it

                    # Do the bottom-half space projection of subject box and the object bbox overlap?
                    # fetch PostGis relations between bounding box pair
                    overlaps,touches = compute_spatial_op(self.cursor, rel_id)
                    if not (overlaps or touches):
                        pred = 'near' # generalises to "near" because not strictly "on"

                if pred in alias_index["in"]:
                    continue
                # 2. update VG predicate statistics
                VG_stats = update_VG_stats(VG_stats, pred, aliases,alias_index,sub_syn, obj_syn)

        # save stats locally
        with open(self.path_to_VGstats, 'w') as fout:
            json.dump(VG_stats,fout)
        print("Data preparation complete... took %f seconds" % (time.time() - start))
        return VG_stats

    def query_conceptnet_web(self,term1,term2,base='/c/en/'):
        node = requests.get('http://api.conceptnet.io/relatedness?node1=' + \
                                base + term1 + '&node2=' + base + term2).json()
        return node['value']