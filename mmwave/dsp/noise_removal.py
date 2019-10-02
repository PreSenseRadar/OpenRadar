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


def peak_grouping_along_doppler(det_obj_2d,
                                det_matrix,
                                num_doppler_bins):
    """Perform peak grouping along the doppler direction only.
    This is a temporary remedy for the slow and old implementation of peak_grouping_qualified() function residing in
    dsp.py currently. Will merge this back to there to enable more generic peak grouping.
    """
    num_det_objs = det_obj_2d.shape[0]
    range_idx = det_obj_2d['rangeIdx']
    doppler_idx = det_obj_2d['dopplerIdx']
    kernel = np.zeros((num_det_objs, 3), dtype=np.float32)
    kernel[:, 0] = det_matrix[range_idx, doppler_idx - 1]
    kernel[:, 1] = det_obj_2d['peakVal'].astype(np.float32)
    kernel[:, 2] = det_matrix[range_idx, (doppler_idx + 1) % num_doppler_bins]
    detectedFlag = (kernel[:, 1] > kernel[:, 0]) & (kernel[:, 1] > kernel[:, 2])
    return det_obj_2d[detectedFlag]


def range_based_pruning(det_obj_2d_raw,
                        snr_thresh,
                        peak_val_thresh,
                        max_range,
                        min_range,
                        range_resolution):
    """Filter out the objects out of the range and not sufficing SNR/peakVal requirement.

    Filter out the objects based on the two following conditions:
    1. Not within [min_range and max_range].
    2. Does not satisfy SNR/peakVal requirement, where it requires higher standard when closer and lower when further.
    """
    det_obj_2d = det_obj_2d_raw[(det_obj_2d_raw['rangeIdx'] >= min_range) & \
                                (det_obj_2d_raw['rangeIdx'] <= max_range)]
    snr_idx1 = (det_obj_2d['SNR'] > snr_thresh[0, 1]) & (det_obj_2d['rangeIdx'] * range_resolution < snr_thresh[0, 0])
    snr_idx2 = (det_obj_2d['SNR'] > snr_thresh[1, 1]) & \
              (det_obj_2d['rangeIdx'] * range_resolution < snr_thresh[1, 0]) & \
              (det_obj_2d['rangeIdx'] * range_resolution >= snr_thresh[0, 0])
    snr_idx3 = (det_obj_2d['SNR'] > snr_thresh[2, 1]) & (det_obj_2d['rangeIdx'] * range_resolution > snr_thresh[1, 0])
    snr_idx = snr_idx1 | snr_idx2 | snr_idx3

    peak_val_idx = np.logical_not((det_obj_2d['peakVal'] < peak_val_thresh[0, 1]) & \
                                (det_obj_2d['rangeIdx'] * range_resolution < peak_val_thresh[0, 0]))
    combined_idx = snr_idx & peak_val_idx
    det_obj_2d = det_obj_2d[combined_idx]

    return det_obj_2d


def prune_to_peaks(det_obj2_d_raw,
                   det_matrix,
                   num_doppler_bins,
                   reserve_neighbor=False):
    """Reduce the CFAR detected output to local peaks.

    Reduce the detected output to local peaks. If reserveNeighbor is toggled, will also return the larger neighbor. For
    example, given an array [2, 1, 5, 3, 2], default method will return [2, 5] while reserve neighbor will return
    [2, 5, 3]. The neighbor has to be a larger neighbor of the two immediate ones and also be part of the peak. the 1st
    element "1" in the example is not returned because it's smaller than both sides so that it is not part of the peak.

    Args:
        det_obj2_d_raw (np.ndarray): The detected objects structured array which contains the range_idx, doppler_idx,
         peakVal and SNR, etc.
        det_matrix (np.ndarray): Output of doppler FFT with virtual antenna dimensions reduced. It has the shape of
            (num_range_bins, num_doppler_bins).
        num_doppler_bins (int): Number of doppler bins.
        reserve_neighbor (boolean): if toggled, will return both peaks and the larger neighbors.

    Returns:
        cfar_det_obj_index_pruned (np.ndarray): Pruned version of cfar_det_obj_index.
        cfar_det_obj_SNR_pruned (np.ndarray): Pruned version of cfar_det_obj_SNR.
    """

    range_idx = det_obj2_d_raw['rangeIdx']
    doppler_idx = det_obj2_d_raw['dopplerIdx']
    next_idx = doppler_idx + 1
    next_idx[doppler_idx == num_doppler_bins - 1] = 0
    prev_idx = doppler_idx - 1
    prev_idx[doppler_idx == 0] = num_doppler_bins - 1

    prev_val = det_matrix[range_idx, prev_idx]
    current_val = det_matrix[range_idx, doppler_idx]
    next_val = det_matrix[range_idx, next_idx]

    if reserve_neighbor:
        next_next_idx = next_idx + 1
        next_next_idx[next_idx == num_doppler_bins - 1] = 0
        prev_prev_idx = prev_idx - 1
        prev_prev_idx[prev_idx == 0] = num_doppler_bins - 1

        prev_prev_val = det_matrix[range_idx, prev_prev_idx]
        next_next_val = det_matrix[range_idx, next_next_idx]
        is_neighbor_of_peak_next = (current_val > next_next_val) & (current_val > prev_val)
        is_neighbor_of_peak_prev = (current_val > prev_prev_val) & (current_val > next_val)

        pruned_idx = (current_val > prev_val) & (current_val > next_val) | is_neighbor_of_peak_next | is_neighbor_of_peak_prev
    else:
        pruned_idx = (current_val > prev_val) & (current_val > next_val)

    det_obj2_d_pruned = det_obj2_d_raw[pruned_idx]

    return det_obj2_d_pruned
