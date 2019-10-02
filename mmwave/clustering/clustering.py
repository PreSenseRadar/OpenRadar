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
from sklearn.cluster import DBSCAN


def associate_clustering(new_cluster,
                         pre_cluster,
                         max_num_clusters,
                         epsilon,
                         v_factor,
                         use_elevation=False):
    """Associate pre-existing clusters and the new clusters.

    The function performs an association between the pre-existing clusters and the new clusters, with the intent that the
    cluster sizes are filtered.

    Args:
        new_cluster:
        pre_cluster:
        max_num_clusters:
        epsilon:
        v_factor:
        use_elevation:
    """
    num_cluster = max_num_clusters if pre_cluster.shape[0] > max_num_clusters else pre_cluster.shape[0]
    pre_avg_vel = np.expand_dims(pre_cluster[num_cluster]['avgVel'], 0)
    pre_location = pre_cluster[num_cluster]['location']

    new_avg_vel = np.expand_dims(new_cluster[num_cluster]['avgVel'], 1)
    new_location = new_cluster[num_cluster]['location']

    # State is previous cluster. output is output cluster.

    # Check if velocity is close.
    # Modify the velocity threshold if the original speed is smaller than threshold itself.
    v_factors = np.ones_like(new_avg_vel) * v_factor
    v_factors = np.minimum(v_factors, new_avg_vel)
    # Put new_cluster as column vector and pre_cluster as row vector to generate difference matrix.
    vel_diff_mat = np.abs(pre_avg_vel - new_avg_vel)
    closest_vel_idx = np.argmin(vel_diff_mat, axis=1)
    closest_vel_val = vel_diff_mat.min(axis=1)

    # Check if position is close enough
    closest_loc = np.zeros_like(len(new_location))
    for i, new_loc in enumerate(new_location):
        loc_diff = (new_loc[0] - pre_location[:, 0]) ** 2 + \
                   (new_loc[1] - pre_location[:, 1]) ** 2 + \
                   (new_loc[2] - pre_location[:, 2]) ** 2 * use_elevation
        closest_loc[i] = np.argmin(loc_diff, axis=1)

    # Get where both velocity and location are satisfied, boolean mask.
    assoc_idx = (closest_vel_val < v_factors) & (closest_loc < epsilon ** 2)
    # Get the actual index. Value j at index i means that pre_cluster[i] is associated to new_cluster[j].
    # if the value j is -1, it means it didn't find any association.
    assoc_idx = (closest_vel_idx + 1) * assoc_idx - 1

    assoc_flag = np.zeros_like(pre_cluster)
    for i, assoc in enumerate(assoc_idx):
        # If there is an associated cluster and not occupied
        if assoc != -1 and not assoc_flag[i]:
            pre_cluster[i] = new_cluster[assoc]
            pre_cluster['size'] *= 0.875  # IIR filter the size so it won't change rapidly.
        # if this is a new cluster.
        elif assoc != -1:
            np.append(pre_cluster, new_cluster[assoc])
        # if the associated new cluster is occupied.
        else:
            continue

    return pre_cluster


def radar_dbscan(det_obj_2d, weight, doppler_resolution, use_elevation=False):
    """DBSCAN for point cloud. Directly call the scikit-learn.dbscan with customized distance metric.

    DBSCAN algorithm for clustering generated point cloud. It directly calls the dbscan from scikit-learn but with
    customized distance metric to combine the coordinates and weighted velocity information.

    Args:
        det_obj_2d (ndarray): Numpy array containing the rangeIdx, dopplerIdx, peakVal, xyz coordinates of each detected
            points. Can have extra SNR entry, not necessary and not used.
        weight (float): Weight for velocity information in combined distance metric.
        doppler_resolution (float): Granularity of the doppler measurements of the radar.
        use_elevation (bool): Toggle to use elevation information for DBSCAN and output clusters.

    Returns:
        clusters (np.ndarray): Numpy array containing the clusters' information including number of points, center and
            size of the clusters in x,y,z coordinates and average velocity. It is formulated as the structured array for
            numpy.
    """

    # epsilon defines max cluster width
    custom_distance = lambda obj1, obj2: \
        (obj1[3] - obj2[3]) ** 2 + \
        (obj1[4] - obj2[4]) ** 2 + \
        use_elevation * (obj1[5] - obj2[5]) ** 2 + \
        weight * ((obj1[1] - obj2[1]) * doppler_resolution) ** 2

    labels = DBSCAN(eps=1.25, min_samples=1, metric=custom_distance).fit_predict(det_obj_2d)
    unique_labels = sorted(
        set(labels[labels >= 0]))  # Exclude the points clustered as noise, i.e, with negative labels.
    dtype_location = '(' + str(2 + use_elevation) + ',)<f4'
    dtype_clusters = np.dtype({'names': ['num_points', 'center', 'size', 'avgVelocity'],
                               'formats': ['<u4', dtype_location, dtype_location, '<f4']})
    clusters = np.zeros(len(unique_labels), dtype=dtype_clusters)
    for label in unique_labels:
        clusters['num_points'][label] = det_obj_2d[label == labels].shape[0]
        clusters['center'][label] = np.mean(det_obj_2d[label == labels, 3:6], axis=0)[:(2 + use_elevation)]
        clusters['size'][label] = np.amax(det_obj_2d[label == labels, 3:6], axis=0)[:(2 + use_elevation)] - \
                                  np.amin(det_obj_2d[label == labels, 3:6], axis=0)[:(2 + use_elevation)]
        clusters['avgVelocity'][label] = np.mean(det_obj_2d[:, 1], axis=0) * doppler_resolution

    return clusters
