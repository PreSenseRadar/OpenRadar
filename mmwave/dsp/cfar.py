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
from scipy.ndimage import convolve1d
from .utils import *

""" Various cfar algorithm types

From https://www.mathworks.com/help/phased/ug/constant-false-alarm-rate-cfar-detectors.html
|-----------------------------------------------------------------------------------------------------------|
|   Algorithm                       |   Typical Usage                                                       |
|-----------------------------------------------------------------------------------------------------------|
|   Cell-averaging CFAR             |   Most situations                                                     |
|   Greatest-of cell-averaging CFAR |   When it is important to avoid false alarms at the edge of clutter   |
|   Smallest-of cell-averaging CFAR |   When targets are closely located                                    |
|   Order statistic CFAR            |   Compromise between greatest-of and smallest-of cell averaging       |
|-----------------------------------------------------------------------------------------------------------|

"""

CFAR_CA = 1
CFAR_CASO = 2
CFAR_CAGO = 3


def cell_average_wrap(arr, *argv, **kwargs):
    """Wrapper function around the cell_average_wrap_threshold. Created per TI's implementation.

    Note:
        May deprecate in the future.

    Args:
        arr (list or ndarray): Noisy array to perform cfar on with log values
        *argv: See cfar.cell_average_wrap_threshold
        **kwargs: See cfar.cell_average_wrap_threshold

    Returns:
        (ndarray): Bit mask of detected peaks

    """
    if isinstance(arr, list):
        arr = np.array(arr)
    threshold, _ = cell_average_wrap_threshold(arr, *argv, **kwargs)
    ret = (arr > threshold)
    return ret


def cell_average_wrap_threshold(arr, l_bound=4000, guard_len=4, noise_len=8):
    """Perform CFAR-CA detection on the input array.

    Args:
        arr (list or ndarray): Noisy array to perform CFAR. The arr is expected bo be 1d list or array and with \
                                log values.
        l_bound (int): Additive lower bound of detection threshold.
        guard_len (int): Left and right side guard samples for leakage protection.
        noise_len (int): Left and right side noise samples after guard samples.

    Returns:
        threshold (ndarray): CFAR generated threshold based on inputs (Peak detected if arr[i] > threshold[i]) \
                                for designated false-positive rate
        noise_floor (ndarray): noise values with the same shape as input arr.
        
    """
    if isinstance(arr, list):
        arr = np.array(arr)
    assert type(arr) == np.ndarray

    kernel = np.ones(1 + (2 * guard_len) + (2 * noise_len), dtype=arr.dtype) / (2 * noise_len)
    kernel[noise_len:noise_len + (2 * guard_len) + 1] = 0

    noise_floor = convolve1d(arr, kernel, mode='wrap')
    threshold = noise_floor + l_bound

    return threshold, noise_floor


def ca_so_go(arr, *argv, **kwargs):
    """Performs non-wrapping cfar detection on the input array with adjustable methods.

    Note:
        May be separated into multiple functions at a later time

    Args:
        arr (list or ndarray:
        *argv: See cfar.ca_so_go_threshold
        **kwargs: See cfar.ca_so_go_threshold

    Returns:
        ret (ndarray): Bit mask of size len(arr) with detected objects from arr

    """
    if isinstance(arr, list):
        arr = np.array(arr)
    threshold = ca_so_go_threshold(arr, *argv, **kwargs)
    ret = (arr > threshold)
    return ret


def ca_so_go_threshold(arr, r_shift=4, l_bound=5600,
                       guard_len=4, noise_len=8,
                       cfar_type=CFAR_CA):
    """Performs non-wrapping cfar detection on the input array with adjustable methods

    Args:
        arr (list or array): Noisy array to perform cfar on with log values
        r_shift (int): 1/2^r_shift detection threshold fraction from sum
        l_bound (int): Additive lower bound of detection threshold
        guard_len (int): Left and right side guard samples for leakage protection
        noise_len (int): Left and right side noise samples after guard samples
        cfar_type (int): Algorithm variant to create adaptive threshold
        
    Returns:
        threshold (np.ndarray): CFAR generated threshold based on inputs (Peak detected if arr[i] > threshold[i])
        
    """
    if isinstance(arr, list):
        arr = np.array(arr)
    assert type(arr) == np.ndarray

    if cfar_type == CFAR_CA:
        kernel = np.ones(1 + (2 * guard_len) + (2 * noise_len), dtype=arr.dtype)
        kernel[noise_len:noise_len + (2 * guard_len) + 1] = 0

        # MODIFIED FROM BASE ALGORITHM
        threshold = convolve1d(arr, kernel, mode='constant')

        # Adaptive Average
        avg_arr = np.zeros_like(threshold)
        avg_arr[guard_len + 1:guard_len + noise_len + 1] += np.arange(1, noise_len + 1, dtype=arr.dtype)
        avg_arr[guard_len + noise_len + 1:-(guard_len + noise_len) - 1] = 2 * noise_len
        avg_arr[-(guard_len + noise_len) - 1:-guard_len - 1] += np.arange(noise_len, 0, -1, dtype=arr.dtype)
        avg_arr[avg_arr < (2 * noise_len)] += noise_len
        threshold = threshold / avg_arr
        threshold = threshold + l_bound

        return threshold

    n = len(arr)
    threshold = np.zeros(n, dtype=arr.dtype)
    cut_idx = 0

    # First portion
    while cut_idx < (guard_len + noise_len):
        right_idx = (cut_idx + 1) + guard_len
        sum_right = np.sum(arr[right_idx:right_idx + noise_len])

        sum_total = sum_right

        threshold[cut_idx] = (sum_total >> (r_shift - 1)) + l_bound
        cut_idx += 1

    # Middle portion
    while cut_idx < (n - (guard_len + noise_len)):
        left_idx = cut_idx - (guard_len + noise_len)
        sum_left = np.sum(arr[left_idx:left_idx + noise_len])

        right_idx = (cut_idx + 1) + guard_len
        sum_right = np.sum(arr[right_idx:right_idx + noise_len])

        # sum_total = None
        if cfar_type == CFAR_CA:
            sum_total = sum_left + sum_right
            threshold[cut_idx] = (sum_total >> r_shift) + l_bound
        elif cfar_type == CFAR_CAGO:
            sum_total = sum_left if sum_left > sum_right else sum_right
            threshold[cut_idx] = (sum_total >> (r_shift - 1)) + l_bound
        elif cfar_type == CFAR_CASO:
            sum_total = sum_left if sum_left < sum_right else sum_right
            threshold[cut_idx] = (sum_total >> (r_shift - 1)) + l_bound
        else:
            print('Unknown cfar_type')

        cut_idx += 1

    # Last portion
    while cut_idx < n:
        left_idx = cut_idx - (guard_len + noise_len)
        sum_left = np.sum(arr[left_idx:left_idx + noise_len])

        sum_total = sum_left

        threshold[cut_idx] = (sum_total >> (r_shift - 1)) + l_bound
        cut_idx += 1

    return threshold


def ordered_statistics_wrap(arr, *argv, **kwargs):
    """Performs non-wrapping cfar detection on the input array with adjustable methods.

    Args:
        arr (ndarray): Noisy array to perform cfar on with log values
        *argv:
        **kwargs:

    Returns:
        ndarray: Bit mask of size len(arr) with detected objects from arr

    """
    if isinstance(arr, list):
        arr = np.array(arr)
    threshold = ordered_statistics_wrap_threshold(arr, *argv, **kwargs)
    ret = (arr > threshold)
    return ret


def ordered_statistics_wrap_threshold(arr, k=12, prob_fa=None, noise_len=8, scale=1.1):
    """Performs non-wrapping cfar detection on the input array with adjustable methods
            
    Args:
        arr (list or ndarray): Noisy array to perform cfar on with log values
        k (int): Ordered statistic rank to sample from
        prob_fa (float): Probability of false alarm, used to calculate scale
        noise_len (int): Left and right side noise samples after guard samples
        scale (int): Scaling factor, if not None prob_fa parameter is ignored
    
    Returns: 
        threshold (ndarray): Bit mask of size len(arr) with detected objects from arr

    TODO: prob_fa is buggy

    """
    if isinstance(arr, list):
        arr = np.array(arr, dtype=np.uint32)

    if not scale:
        scale = np.log(1 / prob_fa)

    n = len(arr)
    threshold = np.zeros(n, dtype=np.float32)
    cut_idx = 0

    # Initial CUT
    left_idx = list(np.arange(n - noise_len, n))
    right_idx = list(np.arange(1, 1 + noise_len))
    window = np.append(arr[left_idx], arr[right_idx])
    window.partition(k)
    threshold[cut_idx] = window[k] * scale

    # All other CUTs
    while cut_idx < n:
        left_idx.pop(0)
        left_idx.append((cut_idx - 1) % n)

        right_idx.pop(0)
        right_idx.append((cut_idx + noise_len) % n)

        window = np.append(arr[left_idx], arr[right_idx])
        window.partition(k)
        threshold[cut_idx] = window[k] * scale

        cut_idx += 1

    return threshold


WRAP_UP_LIST_IDX = lambda x, total: x if x >= 0 else x + total
WRAP_DN_LIST_IDX = lambda x, total: x if x < total else x - total
WRAP_DOPPLER_IDX = lambda x, num_doppler_bins: np.bitwise_and(x, num_doppler_bins - 1)
DOPPLER_IDX_TO_SIGNED = lambda idx, fft_size: idx if idx < fft_size // 2 else idx - fft_size


def peak_grouping(obj_raw,
                  det_matrix,
                  num_doppler_bins,
                  max_range_idx,
                  min_range_idx,
                  group_in_doppler_direction,
                  group_in_range_direction):
    """Performs peak grouping on detection Range/Doppler matrix.
     
    The function groups neighboring peaks into one. The grouping is done according to two input flags: 
    group_in_doppler_direction and group_in_doppler_direction. For each detected peak the function checks if the peak is
    greater than its neighbors. If this is true, the peak is copied to the output list of detected objects. The 
    neighboring peaks that are used for checking are taken from the detection matrix and copied into 3x3 kernel
    regardless of whether they are CFAR detected or not. Note: Function always reads 9 samples per detected object 
    from L3 memory into local array tempBuff, but it only needs to read according to input flags. For example if only
    the group_in_doppler_direction flag is set, it only needs to read middle row of the kernel, i.e. 3 samples per
    target from detection matrix.
    
    Args:
        obj_raw (np.ndarray): (num_detected_objects, 3). detected objects from CFAR.
        det_matrix (np.ndarray): Range-doppler profile. shape is numRangeBins x num_doppler_bins.
        num_doppler_bins (int): number of doppler bins.
        max_range_idx (int): max range of detected objects.
        min_range_idx (int): min range of detected objects
        group_in_doppler_direction (int): flag to perform grouping along doppler direction.
        group_in_range_direction (int): flag to perform grouping along range direction.
        
    Returns:
        obj_out (np.ndarray):  detected object after grouping.
        
    """

    num_detected_objects = obj_raw.shape[0]

    num_obj_out = 0
    kernel = np.empty([9])

    if (group_in_doppler_direction == 1) and (group_in_range_direction == 1):
        # Grouping both in Range and Doppler direction
        start_ind = 0
        step_ind = 1
        end_ind = 8
    elif (group_in_doppler_direction == 0) and (group_in_range_direction == 1):
        # Grouping only in Range direction
        start_ind = 1
        step_ind = 3
        end_ind = 7
    elif (group_in_doppler_direction == 1) and (group_in_range_direction == 0):
        # Grouping only in Doppler direction */
        start_ind = 3
        step_ind = 1
        end_ind = 5
    else:
        # No grouping, copy all detected objects to the output matrix within specified min max range
        # num_detected_objects = min(num_detected_objects, MAX_OBJ_OUT)
        obj_out = obj_raw[obj_raw[:, RANGEIDX] <= max_range_idx and obj_raw[:, RANGEIDX] > min_range_idx]
        obj_out[:, DOPPLERIDX] = np.bitwise_and(obj_out[:, DOPPLERIDX], num_doppler_bins - 1)

        return obj_out

    # Start checking
    obj_out = np.zeros((num_obj_out, 3))
    for i in range(num_detected_objects):
        detected_obj_flag = 0
        range_idx = obj_raw[i, 0]
        doppler_idx = obj_raw[i, 1]
        peak_val = obj_raw[i, 2]

        if (range_idx <= max_range_idx) and (range_idx >= min_range_idx):
            detected_obj_flag = 1

            # Fill local 3x3 kernel from detection matrix in L3
            start_idx = (range_idx - 1) * num_doppler_bins
            temp_ptr = det_matrix[start_idx:]
            row_start = 0
            row_end = 2

            if range_idx == min_range_idx:
                start_idx = range_idx * num_doppler_bins
                temp_ptr = det_matrix[start_idx:]
                row_start = 1
                kernel[0] = 0
                kernel[1] = 0
                kernel[2] = 0
            elif range_idx == max_range_idx:
                row_end = 1
                kernel[6] = 0
                kernel[7] = 0
                kernel[8] = 0

            for j in range(row_start, row_end + 1):
                for k in range(3):

                    temp_idx = doppler_idx + (k - 1)

                    if temp_idx < 0:
                        temp_idx += num_doppler_bins
                    elif temp_idx >= num_doppler_bins:
                        temp_idx -= num_doppler_bins

                    kernel[j * 3 + k] = temp_ptr[temp_idx]

                temp_ptr = temp_ptr[num_doppler_bins:]

            # Compare the detected object to its neighbors
            # Detected object is at index 4
            for k in range(start_ind, end_ind + 1, step_ind):
                if kernel[k] > kernel[4]:
                    detected_obj_flag = 0

        if detected_obj_flag == 1:
            obj_out[num_obj_out, 0] = range_idx
            obj_out[num_obj_out, 1] = DOPPLER_IDX_TO_SIGNED(doppler_idx, num_doppler_bins)
            obj_out[num_obj_out, 2] = peak_val
            num_obj_out += 1

        if num_obj_out >= MAX_OBJ_OUT:
            break

    return num_obj_out, obj_out


def peak_grouping_qualified(obj_raw,
                            num_doppler_bins,
                            max_range_idx,
                            min_range_idx,
                            group_in_doppler_direction,
                            group_in_range_direction):
    """Performs peak grouping on list of CFAR detected objects.
     
    The function groups neighboring peaks into one. The grouping is done according to two input flags: 
    group_in_doppler_direction and group_in_doppler_direction. For each detected peak the function checks if the peak is
    greater than its neighbors. If this is true, the peak is copied to the output list of detected objects. The 
    neighboring peaks that are used for checking are taken from the list of CFAR detected objects, (not from the 
    detection matrix), and copied into 3x3 kernel that has been initialized to zero for each peak under test. If the 
    neighboring cell has not been detected by CFAR, its peak value is not copied into the kernel. Note: Function always 
    search for 8 peaks in the list, but it only needs to search according to input flags.
    
    Args:
        obj_raw (np.ndarray): (num_detected_objects, 3). detected objects from CFAR.
        num_doppler_bins (int): number of doppler bins.
        max_range_idx (int): max range of detected objects.
        min_range_idx (int): min range of detected objects
        group_in_doppler_direction (int): flag to perform grouping along doppler direction.
        group_in_range_direction (int): flag to perform grouping along range direction.
        
    Returns:
        obj_out (np.ndarray):  detected object after grouping.
        
    """

    num_detected_objects = obj_raw.shape[0]

    if (group_in_doppler_direction == 1) and (group_in_range_direction == 1):
        # Grouping both in Range and Doppler direction
        start_ind = 0
        step_ind = 1
        end_ind = 8
    elif (group_in_doppler_direction == 0) and (group_in_range_direction == 1):
        # Grouping only in Range direction
        start_ind = 1
        step_ind = 3
        end_ind = 7
    elif (group_in_doppler_direction == 1) and (group_in_range_direction == 0):
        # Grouping only in Doppler direction */
        start_ind = 3
        step_ind = 1
        end_ind = 5
    else:
        # No grouping, copy all detected objects to the output matrix within specified min max range
        num_detected_objects = min(num_detected_objects, MAX_OBJ_OUT)
        obj_out = obj_raw[(obj_raw['range_idx'][:num_detected_objects] <= max_range_idx) &
                          (obj_raw['range_idx'][:num_detected_objects] > min_range_idx)]

        return obj_out

    # Start checking
    idx_obj_in_range = np.argwhere((obj_raw['range_idx'] <= max_range_idx) &
                                   (obj_raw['range_idx'] >= min_range_idx))[:, 0]

    obj_in_range = obj_raw[idx_obj_in_range]
    kernels = np.zeros((obj_in_range.shape[0], 9))
    detected_obj_flag = np.ones(obj_in_range.shape[0])

    # Populate the middle column.
    # Populate the 4th element.
    kernels[:, 4] = obj_in_range['peakVal']

    # Populate the 1st element.
    obj_in_range_previous = obj_raw[idx_obj_in_range - 1]
    assert obj_in_range_previous.shape == obj_in_range.shape, "obj_in_range_previous indexing is wrong"
    idx_temp = ((obj_in_range_previous['range_idx']) == (obj_in_range['range_idx'] - 1)) & \
               ((obj_in_range_previous['doppler_idx']) == (obj_in_range['doppler_idx']))
    kernels[idx_temp, 1] = obj_in_range_previous['peakVal'][idx_temp]
    # 0th detected object has no left neighbor.
    kernels[idx_obj_in_range[idx_obj_in_range[:] == 0], 1] = 0

    # Populate the 7th element.
    obj_in_range_next = obj_raw[(idx_obj_in_range + 1) % num_detected_objects]
    assert obj_in_range_next.shape == obj_in_range.shape, "obj_in_range_next indexing is wrong"
    idx_temp = ((obj_in_range_next['range_idx']) == (obj_in_range['range_idx'] + 1)) & \
               ((obj_in_range_next['doppler_idx']) == (obj_in_range['doppler_idx']))
    kernels[idx_temp, 7] = obj_in_range_next['peakVal'][idx_temp]
    # last detected object, i.e. num_detected_objects-th has no left neighbor.
    kernels[idx_obj_in_range[idx_obj_in_range[:] == num_detected_objects], 7] = 0

    for i, idxDeteced in enumerate(idx_obj_in_range):
        doppler_idx = obj_in_range['doppler_idx'][i]
        range_idx = obj_in_range['range_idx'][i]
        # Fill the left column
        k_left = WRAP_UP_LIST_IDX(idxDeteced - 1, num_detected_objects)
        k_right = WRAP_DN_LIST_IDX(idxDeteced + 1, num_detected_objects)
        for _ in range(num_detected_objects):
            k_left_doppler_idx = obj_raw['doppler_idx'][k_left]
            k_left_range_idx = obj_raw['range_idx'][k_left]
            k_left_peak_val = obj_raw['peakVal'][k_left]
            if k_left_doppler_idx == WRAP_DOPPLER_IDX(doppler_idx - 2, num_doppler_bins):
                break
            if k_left_range_idx == range_idx + 1 and k_left_doppler_idx == WRAP_DOPPLER_IDX(doppler_idx - 1,
                                                                                            num_doppler_bins):
                kernels[i, 6] = k_left_peak_val
            elif k_left_range_idx == range_idx and k_left_doppler_idx == WRAP_DOPPLER_IDX(doppler_idx - 1,
                                                                                          num_doppler_bins):
                kernels[i, 3] = k_left_peak_val
            elif k_left_range_idx == range_idx - 1 and k_left_doppler_idx == WRAP_DOPPLER_IDX(doppler_idx - 1,
                                                                                              num_doppler_bins):
                kernels[i, 0] = k_left_peak_val
            k_left = WRAP_UP_LIST_IDX(k_left - 1, num_detected_objects)

            k_right_doppler_idx = obj_raw['doppler_idx'][k_right]
            k_right_range_idx = obj_raw['range_idx'][k_right]
            k_right_peak_val = obj_raw['peakVal'][k_right]
            if k_right_doppler_idx == WRAP_DOPPLER_IDX(doppler_idx - 2, num_doppler_bins):
                break
            if k_right_range_idx == range_idx + 1 and k_right_doppler_idx == WRAP_DOPPLER_IDX(doppler_idx + 1,
                                                                                              num_doppler_bins):
                kernels[i, 8] = k_right_peak_val
            elif k_right_range_idx == range_idx and k_right_doppler_idx == WRAP_DOPPLER_IDX(doppler_idx + 1,
                                                                                            num_doppler_bins):
                kernels[i, 5] = k_right_peak_val
            elif k_right_range_idx == range_idx - 1 and k_right_doppler_idx == WRAP_DOPPLER_IDX(doppler_idx + 1,
                                                                                                num_doppler_bins):
                kernels[i, 2] = k_right_peak_val
            k_right = WRAP_DN_LIST_IDX(k_right + 1, num_detected_objects)

    detected_obj_flag[np.argwhere(np.max(kernels[:, start_ind:end_ind:step_ind]) != kernels[:, 4])] = 0
    obj_out = obj_in_range[detected_obj_flag[:] == 1]

    if obj_out.shape[0] > MAX_OBJ_OUT:
        obj_out = obj_out[:MAX_OBJ_OUT, ...]

    return obj_out


