"""KB class"""
import time
from re import search

from PostGIS import *
from VG import *


class KnowledgeBase():
    def __init__(self, args):
        # Load raw external KB data
        self.path_to_VGrel = os.path.join(args.path_to_data, 'relationships.json')
        self.path_to_VGstats = os.path.join(args.path_to_data, 'VG_spatial_stats.json')
        self.path_to_predicate_aliases = os.path.join(args.path_to_data, 'relationship_aliases.txt')
        self.predicate_set = [("in", "inside"), ("on",),  ("on top of",), ("against",),
                              ("front",), ("behind",), ("left",), ("right",),
                              ("next to","beside","adjacent", "on side of"),  ("under", "below"),
                              ("above",), ("near","by","around")] #predicate pool, with synonyms, needed to derive the commonsense predicates of (Landau & Jackendoff 1993)
                                                    # also adopted in our paper
        #if predicate contains any of the above but not also any of the others

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
        print("Preparing spatial data first. May take a while...")
        start = time.time()
        # Load VG raw relationships and aliases
        raw_data = load_rel_bank(self) #, alias_index
        # flatten alias index
        #alias_mergedlist = list(alias_index.values())

        VG_stats = {k: {} for k in ["predicates", "subjects", "objects"]}
        for entry in raw_data: # entry = image
            img_rels = entry["relationships"]  # all relationships for a given image

            for rel in img_rels:
                pred = rel['predicate'].lower()
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

                # for predicates we need aliases (if any is found) because synsets can be unknown or too generic
                # Find closest match between pred and predicate set, if any

                hits = [p_ for p in self.predicate_set for p_ in p if search(pred,p_) is not None]
                if len(hits) == 0:
                    print("Predicate %s not similar to any in the list" % pred)
                    continue #otherwise skip

                # TODO find most exact/relevant match
                match = ''
                if match =='left':
                    #TODO check how it is handled in the below methods
                    VG_stats = update_VG_stats(VG_stats, "leftOf", sub_syn, obj_syn)
                    # union of right and left also counts as beside
                    VG_stats = update_VG_stats(VG_stats, "beside", sub_syn, obj_syn)
                elif match == 'right':
                    VG_stats = update_VG_stats(VG_stats, "rightOf", sub_syn, obj_syn)
                    # union of right and left also counts as beside
                    VG_stats = update_VG_stats(VG_stats, "beside", sub_syn, obj_syn)
                elif match == 'front':
                    VG_stats = update_VG_stats(VG_stats, "inFrontOf", sub_syn, obj_syn)
                elif match =='behind' or match=='above' or match=='below' or match =='on top of' or match=='in'\
                        or match=='against' or match=='beside' or match=='near':
                    VG_stats = update_VG_stats(VG_stats, match, sub_syn, obj_syn) #no need to reformat QSR name
                elif match =='under':
                    VG_stats = update_VG_stats(VG_stats, "below", sub_syn, obj_syn)
                elif match =="inside":
                    VG_stats = update_VG_stats(VG_stats, "in", sub_syn, obj_syn)
                elif match in ("next to","adjacent", "on side of"):
                    VG_stats = update_VG_stats(VG_stats, "beside", sub_syn, obj_syn)
                elif match in ("by","around"):
                    VG_stats = update_VG_stats(VG_stats, "near", sub_syn, obj_syn)

                elif match =='on':
                    #TODO disambiguate on top of from against (i.e., seen as union of leanson and affixed on)
                    pass
                """
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

                # 2. update VG predicate statistics
                VG_stats = update_VG_stats(VG_stats, pred, aliases,alias_index,sub_syn, obj_syn)
                """
        # save stats locally
        with open(self.path_to_VGstats, 'w') as fout:
            json.dump(VG_stats,fout)
        print("Data preparation complete... took %f seconds" % (time.time() - start))
        return VG_stats

