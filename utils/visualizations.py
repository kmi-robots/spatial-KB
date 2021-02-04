"""
Matplotlib visualization of 3D polyhedral surfaces
not rendered in PGAdmin/PostGRE SQL viewer
"""
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx



def cuboid_data(o, size=(1,1,1)):
    # generic definition of cuboid of edge 1
    X = [[[0, 1, 0], [0, 0, 0], [1, 0, 0], [1, 1, 0]],
         [[0, 0, 0], [0, 0, 1], [1, 0, 1], [1, 0, 0]],
         [[1, 0, 1], [1, 0, 0], [1, 1, 0], [1, 1, 1]],
         [[0, 0, 1], [0, 0, 0], [0, 1, 0], [0, 1, 1]],
         [[0, 1, 0], [0, 1, 1], [1, 1, 1], [1, 1, 0]],
         [[0, 1, 1], [0, 0, 1], [1, 0, 1], [1, 1, 1]]]
    X = np.array(X).astype(float)
    for i in range(3): # multiply by given input size
        X[:,:,i] *= size[i]
    X += np.array(o) # shift based on origin coordinates

    return X


def plotCubeAt2(positions, sizes=None, colors=None, **kwargs):
    # if colors not given generates them randomly
    if not isinstance(colors, (list, np.ndarray)): colors = ["C0"] * len(positions)
    # if sizes not given reverts back to cuboid of size 1
    if not isinstance(sizes, (list, np.ndarray)): sizes = [(1, 1, 1)] * len(positions)
    g = []
    for p, s, c in zip(positions, sizes, colors):
        g.append(cuboid_data(p, size=s))

    return Poly3DCollection(np.concatenate(g),
                            facecolors=np.repeat(colors, 6), **kwargs)


positions = [(0,0,0), (1,7,1),(-5,-5,2)]
sizes = [(4,5,3), (3,3,7), (1,1,1)]
colors = ["crimson","limegreen","blue"]

fig = plt.figure()
ax = fig.gca(projection='3d')
pc = plotCubeAt2(positions,sizes,colors=colors, edgecolor="k")
ax.add_collection3d(pc)
ax.set_xlim([-5,6])
ax.set_ylim([-5,13])
ax.set_zlim([-3,9])

plt.show()

