"""
Computes halfspace projections of 3D bounding boxes
in spatial DB
Returns QSR based on halfspace projections
"""

def compute_hs_projections(box_dict, s=3):
    # As in Deeken et al., (2018) compute halfspace 3D projections of 3D boxes
    # , i.e., 3D extrusions, obtained by multiplying the extent by a scaling factor (s)
    # returns minmax of each projection
    for object_id in box_dict.keys():
        x = box_dict[object_id]["x_extent"]
        y = box_dict[object_id]["y_extent"]
        z = box_dict[object_id]["z_extent"]

        xmin_, ymin_, zmin_, xmax_,ymax_, zmax_= box_dict[object_id]["minmax"]
        box_dict[object_id]["top_hs"] = [xmin_, ymin_, zmax_,xmax_, ymax_,(zmax_+z*s)]
        box_dict[object_id]["btm_hs"] = [xmin_, ymin_, (zmin_-z*s),xmax_,ymax_,zmin_]
        box_dict[object_id]["back_hs"] =[xmax_,ymin_, zmin_,(xmax_+x*s), ymax_,zmax_]
        box_dict[object_id]["front_hs"] =[(xmin_-x*s),ymin_, zmin_,xmin_,ymax_,zmax_]
        box_dict[object_id]["left_hs"] =[xmin_,ymax_,zmin_,xmax_,(ymax_+y*s),zmax_]
        box_dict[object_id]["right_hs"] =[xmin_,(ymin_-y*s),zmin_,xmax_,ymin_,zmax_]

    return box_dict

