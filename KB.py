"""KB class"""
import json
import os
from nltk.corpus import wordnet as wn
from collections import Counter

from onto import init_onto
from PostGIS import connect_DB, disconnect_DB

class KnowledgeBase():
    def __init__(self, args):
        self.ontology = init_onto(args.IRI)
        # Load raw external KB data
        self.path_to_VGrel = os.path.join(args.path_to_data, 'relationships.json')
        self.path_to_VGstats = os.path.join(args.path_to_data, 'VG_stats.json')
        self.path_to_predicate_aliases = os.path.join(args.path_to_data, 'relationship_aliases.txt')
        self.db_user = os.environ['USER']
        self.dbname = args.dbname

    def db_session(self):
        # Open connection
        connection, cursor = connect_DB(self.db_user,self.dbname)

        # Check if already populated with data
        if not os.path.isfile(self.path_to_VGstats): #data preparation needed
            self.VG_stats = self.data_prep()
        else:
            with open(self.path_to_VGstats) as fin:
                self.VG_stats = json.load(fin)

        #Query db

        #Close connection, to avoid db issues
        if connection is not None:
            disconnect_DB(connection,cursor)

    def data_prep(self):

        # Load VG raw relationships and aliases
        with open(self.path_to_VGrel) as ind, open(self.path_to_predicate_aliases) as aliasin:
            alias_list = aliasin.read().splitlines()[:-1] # last line is empty
            raw_data = json.load(ind)

        alias_index = {}
        for aline in alias_list:
            alist = aline.split(",")
            if alist[0] in alias_index: alias_index[alist[0]].extend(alist[1:])
            else:  alias_index[alist[0]] = alist
        alias_mergedlist = list(alias_index.values())

        VG_stats = {k: {} for k in ["predicates", "subjects", "objects"]}
        for entry in raw_data:
            img_rels = entry["relationships"]  # all relationships for a given image
            for rel in img_rels:
                rel_id = rel['relationship_id']
                pred = rel['predicate'].lower()
                sub_syn = rel['subject']['synsets']
                obj_syn = rel['object']['synsets']
                try:
                    sub = rel['subject']['name']
                    obj = rel['object']['name']
                except KeyError:  # no annotation for subject or object
                    # skip if not even synset is available
                    if len(sub_syn) == 0 or len(obj_syn) == 0: continue

                if len(sub_syn) > 1 or len(obj_syn) > 1 or pred == '' or pred == ' ':
                    # Skipping composite periods
                    # (e.g.  subject: green trees seen  pred: green trees by road object: trees on roadside.)
                    # # ( e.g.  subject: see cupboard  pred: cupboard black object: cupboard not white. )
                    # as well as empty predicates
                    continue
                try:
                    pred_synset = wn.synset(rel['synsets'][0])
                    postag = pred_synset.pos()
                except IndexError:  # synset predicate not specified
                    pred_synset, postag = None, None

                # for predicates we need aliases because they may have no synset associated
                aliases = [alias_set for alias_set in alias_mergedlist if pred in alias_set][0]

                sub_coords = (rel['subject']['x'], rel['subject']['x'] + rel['subject']['w'], \
                              rel['subject']['y'], rel['subject']['y'] + rel['subject']['h'])

                # 1. TODO Populate spatial DB # (All, regardless of which subset has both sub_syn and obj_syn, i.e., we are only interested in polygons and preds)

                # 2. update VG predicate statistics (only for records with univoque sub_syn,obj_syn pair)

                if len(sub_syn) >0 and len(obj_syn)>0:
                    # How many times subj - pred - obj?
                    if pred not in VG_stats["predicates"]:
                        VG_stats["predicates"][pred]["relations"] = Counter()
                        VG_stats["predicates"][pred]["aliases"] = aliases
                    VG_stats["predicates"][pred]["relations"][str((str(sub_syn[0]), str(obj_syn[0])))] +=1

                    # how many times subject in relationship of type pred?
                    if str(sub_syn[0]) not in VG_stats["subjects"]:
                        VG_stats["subjects"][str(sub_syn[0])] = Counter()
                    VG_stats["subjects"][str(sub_syn[0])][pred] += 1

                    # how many times object in relationship of type pred?
                    if str(obj_syn[0]) not in VG_stats["objects"]:
                        VG_stats["objects"][str(obj_syn[0])] = Counter()
                    VG_stats["objects"][str(obj_syn[0])][pred] += 1

        # save stats locally
        with open(self.path_to_VGstats, 'w') as fout:
            json.dump(VG_stats,fout)
        return VG_stats