"""
Computes halfspace projections of 3D bounding boxes
in spatial DB
Returns QSR based on halfspace projections
"""

def compute_hs_projections(box_dict, s=1):
    # As in Deeken et al., (2018) compute halfspace 3D projections of 3D boxes
    # extrusions are obtained by multiplying the extent by a scaling factor (s)
    # returns minmax of each projection
    for object_id in box_dict.keys():
        x = box_dict[object_id]["x_extent"] * s
        y = box_dict[object_id]["y_extent"] * s
        z = box_dict[object_id]["z_extent"] * s
        vs = box_dict[object_id]["vertices"]


        box_dict[object_id]["top_hs"] = [sum(v) for v in zip(vs[3], [-x, -y, 0])] + [sum(v) for v in zip(vs[6], [x, y, z])]
        box_dict[object_id]["btm_hs"] = [sum(v) for v in zip(vs[0], [-x, -y, -z])] + [sum(v) for v in zip(vs[5], [x, y, 0])]
        box_dict[object_id]["front_hs"] = [sum(v) for v in zip(vs[4], [0, -y, -z])] + [sum(v) for v in zip(vs[6], [x, y, z])]
        box_dict[object_id]["back_hs"] = [sum(v) for v in zip(vs[0], [-x, -y, -z])] + [sum(v) for v in zip(vs[2], [0, y, z])]
        box_dict[object_id]["left_hs"] = [sum(v) for v in zip(vs[1], [-x, 0, -z])] + [sum(v) for v in zip(vs[6], [x, y, z])]
        box_dict[object_id]["right_hs"] = [sum(v) for v in zip(vs[0], [-x, -y, -z])] + [sum(v) for v in zip(vs[7], [x, 0, z])]
    return box_dict

