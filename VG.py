"""Visual Genome processing methods"""
import json
from collections import OrderedDict,Counter


def load_rel_bank(KBobj):
    with open(KBobj.path_to_VGrel) as ind, open(KBobj.path_to_predicate_aliases) as aliasin:
        alias_list = aliasin.read().splitlines()[:-1]  # last line is empty
        raw_data = json.load(ind)

    alias_index = OrderedDict()
    for aline in alias_list:
        alist = aline.split(",")
        # also check for duplicate aliases which are not at first position in each line
        match_index = [i for i, vlist in enumerate(alias_index.values()) if alist[0] in vlist]
        if alist[0] in alias_index:
            alias_index[alist[0]].extend(alist[1:])
        elif len(match_index) > 0:
            k = list(alias_index.items())[match_index[0]][0]
            alias_index[k].extend(alist[1:])
        else:
            alias_index[alist[0]] = alist
    return raw_data,alias_index



def update_VG_stats(stats_dict,pred,aliases,alias_index, sub_syn, obj_syn):

    stats_dict = add_relation_counts(stats_dict,pred,aliases, sub_syn, obj_syn)
    # Repeat all of the above for the case of near, i.e., all predicates generalise back to near
    #if pred not in alias_index['near']:
    #    stats_dict = add_relation_counts(stats_dict, "near", alias_index["near"], sub_syn, obj_syn)
    return stats_dict


def add_relation_counts(stats_dict,pred,aliases, sub_syn, obj_syn):
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