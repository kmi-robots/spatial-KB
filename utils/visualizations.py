"""
toy example: Open3d visualization of 3D solids"""
import numpy as np
import open3d as o3d

def minmax2box(minb,maxb, color = (0,0,0)): #line color defaults to black
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

box1 = minmax2box([0,0,0],[4,5,5])
box2 = minmax2box([1,7,1],[4,10,8], color=(0.1,0.9,0.1)) #green
box3 = minmax2box([-5,-5,2],[-4,-4,3], color=(0.,0.,1.)) #blue

mesh_frame = o3d.geometry.TriangleMesh.create_coordinate_frame() # setup coord frame. red = x, green = y, blue = z
o3d.visualization.draw_geometries([box1,box2,box3,mesh_frame]) #visualize results



