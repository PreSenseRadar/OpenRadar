# Copyright 2019 The OpenRadar Authors. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

import numpy as np
from numpy import linalg
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib import cm
import matplotlib.colors as colors


def ellipse_visualize(fig, clusters, points):
    """Visualize point clouds and outputs from 3D-DBSCAN
    
    Args:
        Clusters (np.ndarray): Numpy array containing the clusters' information including number of points, center and size of
                the clusters in x,y,z coordinates and average velocity. It is formulated as the structured array for numpy.
        points (dict): A dictionary that stores x,y,z's coordinates in np arrays
    
    Returns:
        N/A
    """
    # fig = plt.figure(figsize=(8,8))
    ax = fig.add_subplot(111, projection='3d')
    ax.set_zlim3d(bottom=-5, top=5)
    ax.set_ylim(bottom=0, top=10)
    ax.set_xlim(left=-4, right=4)    
    ax.set_xlabel('X Label')
    ax.set_ylabel('Y Label')
    ax.set_zlabel('Z Label')
    # ax.set_aspect('equal')

    # scatter plot
    # ax.scatter(points['x'], points['y'], points['z'])
    # ax.scatter(points[:, 0], points[:, 1], points[:, 2])

    # number of ellipsoids 
    ellipNumber = len(clusters)

    norm = colors.Normalize(vmin=0, vmax=ellipNumber)
    cmap = cm.jet
    m = cm.ScalarMappable(norm=norm, cmap=cmap)

    for indx in range(ellipNumber):
        center = [clusters['center'][indx][0],clusters['center'][indx][1],clusters['center'][indx][2]]

        radii = np.zeros([3,])
        radii[0] = clusters['size'][indx][0]
        radii[1] = clusters['size'][indx][1]
        radii[2] = clusters['size'][indx][2]

        u = np.linspace(0.0, 2.0 * np.pi, 60)
        v = np.linspace(0.0, np.pi, 60)
        x = radii[0] * np.outer(np.cos(u), np.sin(v))
        y = radii[1] * np.outer(np.sin(u), np.sin(v))
        z = radii[2] * np.outer(np.ones_like(u), np.cos(v))


        for i in range(len(x)):
            for j in range(len(x)):
                [x[i,j],y[i,j],z[i,j]] = np.dot([x[i,j],y[i,j],z[i,j]], np.array([[1,0,0],[0,1,0],[0,0,1]])) + center


        ax.plot_surface(x, y, z,  rstride=3, cstride=3,  color=m.to_rgba(indx), linewidth=0.1, alpha=1, shade=True)
        
    plt.show()
