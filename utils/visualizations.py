"""
toy example: Open3d visualization of 3D solids"""
import numpy as np
import open3d as o3d
import copy

def minmax2box(minb,maxb, color = (0.,0.,1.)): #line color defaults to black
    # expects two lists [Xmin, Ymin, Zmin]
    # and [Xmax, Ymax, Zmax]
    minb =[minb] #min bound, size 1 by 3
    minb = np.array(minb).astype(float)
    maxb = [maxb] #max bound
    maxb = np.array(maxb).astype(float)
    c = np.array(color).astype(float).T
    mesh_box = o3d.geometry.AxisAlignedBoundingBox(minb.T,maxb.T)#needs to be 3 by 1
    mesh_box.color = c # fill with solid color
    return mesh_box

def convert_and_rotate(bbox,z_angle=np.pi,center=(0,0,0), color=(0.,0.,0.)):
    R = bbox.get_rotation_matrix_from_xyz((0, 0, z_angle))
    cbox = o3d.geometry.OrientedBoundingBox()
    cbox = cbox.create_from_axis_aligned_bounding_box(copy.deepcopy(bbox))  # rotated bbox
    cbox.color = color
    return cbox.rotate(R,center=False)

box1 = minmax2box([-2,-3,-1],[2,3,1])
box1_r = convert_and_rotate(box1,z_angle=np.pi/5)

#top and btm based on oriented bbox
tophs = minmax2box([-2,-3,1],[2,3,8])#, color=(0.,0.,1.)) #blue
tophs_r = convert_and_rotate(tophs,z_angle=np.pi/5)
btmhs = minmax2box([-2,-3,-8],[2,3,1])
btmhs_r = convert_and_rotate(btmhs,z_angle=np.pi/5)
#other 4 halfspaces based on rotated bbox

backhs = minmax2box([-9,-3,-1],[-2,3,1])#, color=(1.,0.,0.)) #red
#backhs_r = convert_and_rotate(backhs,z_angle=np.pi/5)

fronths = minmax2box([2,-3,-1],[9,3,1])#, color=(0.,1.,0.)) #green
#fronths_r = convert_and_rotate(fronths,z_angle=np.pi/5)
lhs = minmax2box([-2,3,-1],[2,10,1])#, color=(0.,1.,1.)) #acquamarine
#lhs_r = convert_and_rotate(lhs,z_angle=np.pi/5)

rhs = minmax2box([-2,-10,-1],[2,3,1])#, color=(1.,0.,1.)) #fucsia
#rhs_r = convert_and_rotate(rhs,z_angle=np.pi/5)
#box2 = minmax2box([1,7,1],[4,10,8])#, color=(0.,1.,0.)) #green
#box3 = minmax2box([-5,-5,2],[-4,-4,3])#, color=(1.,0.,0.)) #red

mesh_frame = o3d.geometry.TriangleMesh.create_coordinate_frame() # setup coord frame. red = x, green = y, blue = z
robot_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(origin=np.array([-8., 0., 0.])) # setup coord frame. red = x, green = y, blue = z
o3d.visualization.draw_geometries([mesh_frame])
R_ = mesh_frame.get_rotation_matrix_from_xyz((0, 0, np.pi/5))
box_frame = mesh_frame.rotate(R_)
o3d.visualization.draw_geometries([box1_r,robot_frame,box_frame])#, tophs,btmhs, fronths_r, backhs_r, lhs_r, rhs_r, mesh_frame])
#o3d.visualization.draw_geometries([box1,box1_r,tophs,btmhs,fronths_r,backhs_r, mesh_frame])#,box3,mesh_frame]) #visualize results
#o3d.visualization.draw_geometries([box1,lhs, rhs, mesh_frame])#tophs,btmhs,rhs,lhs,fronths, mesh_frame]) #visualize results



