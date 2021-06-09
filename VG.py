"""Visual Genome processing methods"""
import json
from collections import OrderedDict,Counter


def load_rel_bank(KBobj):
    with open(KBobj.path_to_VGrel) as ind:
        raw_data = json.load(ind)
    return raw_data


def update_VG_stats(stats_dict,pred,sub_syn, obj_syn):

    stats_dict = add_relation_counts(stats_dict,pred,sub_syn, obj_syn)
    # Repeat all of the above for the case of near, i.e., all predicates generalise back to near
    #if pred not in alias_index['near']:
    #    stats_dict = add_relation_counts(stats_dict, "near", alias_index["near"], sub_syn, obj_syn)
    return stats_dict

def add_relation_counts(stats_dict,pred,sub_syn, obj_syn):
    # How many times subj - pred - obj?
    if pred not in stats_dict["predicates"]:
        stats_dict["predicates"][pred] = {}
        stats_dict["predicates"][pred]["relations"] = Counter()
        #stats_dict["predicates"][pred]["aliases"] = aliases
    stats_dict["predicates"][pred]["relations"][str((str(sub_syn[0]), str(obj_syn[0])))] += 1

    # how many times subject in relationship of type pred?
    if str(sub_syn[0]) not in stats_dict["subjects"]:
        stats_dict["subjects"][str(sub_syn[0])] = Counter()
    stats_dict["subjects"][str(sub_syn[0])][pred] += 1

    # how many times object in relationship of type pred?
    if str(obj_syn[0]) not in stats_dict["objects"]:
        stats_dict["objects"][str(obj_syn[0])] = Counter()
    stats_dict["objects"][str(obj_syn[0])][pred] += 1
    return stats_dict