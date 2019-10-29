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


def ca(x, *argv, **kwargs):
    """Detects peaks in signal using Cell-Averaging CFAR (CA-CFAR).

    Args:
        x (~numpy.ndarray): Signal.
        *argv: See mmwave.dsp.cfar.ca\_
        **kwargs: See mmwave.dsp.cfar.ca\_

    Returns:
        ~numpy.ndarray: Boolean array of detected peaks in x.

    Examples:
        >>> signal = np.random.randint(100, size=10)
        >>> signal
            array([41, 76, 95, 28, 25, 53, 10, 93, 54, 85])
        >>> det = mm.dsp.ca(signal, l_bound=20, guard_len=1, noise_len=3)
        >>> det
            array([False, False,  True, False, False, False, False,  True, False,
                    True])

        Perform a non-wrapping CFAR

        >>> signal = np.random.randint(100, size=10)
        >>> signal
            array([41, 76, 95, 28, 25, 53, 10, 93, 54, 85])
        >>> det =  mm.dsp.ca(signal, l_bound=20, guard_len=1, noise_len=3, mode='constant')
        >>> det
            array([False,  True,  True, False, False, False, False,  True,  True,
                    True])

    """
    if isinstance(x, list):
        x = np.array(x)
    threshold, _ = ca_(x, *argv, **kwargs)
    ret = (x > threshold)
    return ret


def ca_(x, guard_len=4, noise_len=8, mode='wrap', l_bound=4000):
    """Uses Cell-Averaging CFAR (CA-CFAR) to calculate a threshold that can be used to calculate peaks in a signal.

    Args:
        x (~numpy.ndarray): Signal.
        guard_len (int): Number of samples adjacent to the CUT that are ignored.
        noise_len (int): Number of samples adjacent to the guard padding that are factored into the calculation.
        mode (str): Specify how to deal with edge cells. Examples include 'wrap' and 'constant'.
        l_bound (float or int): Additive lower bound while calculating peak threshold.

    Returns:
        Tuple [ndarray, ndarray]
            1. (ndarray): Upper bound of noise threshold.
            #. (ndarray): Raw noise strength.

    Examples:
        >>> signal = np.random.randint(100, size=10)
        >>> signal
            array([41, 76, 95, 28, 25, 53, 10, 93, 54, 85])
        >>> threshold = mm.dsp.ca_(signal, l_bound=20, guard_len=1, noise_len=3)
        >>> threshold
            (array([70, 76, 64, 79, 81, 91, 74, 71, 70, 79]), array([50, 56, 44, 59, 61, 71, 54, 51, 50, 59]))

        Perform a non-wrapping CFAR thresholding

        >>> signal = np.random.randint(100, size=10)
        >>> signal
            array([41, 76, 95, 28, 25, 53, 10, 93, 54, 85])
        >>> threshold = mm.dsp.ca_(signal, l_bound=20, guard_len=1, noise_len=3, mode='constant')
        >>> threshold
            (array([44, 37, 41, 65, 81, 91, 67, 51, 34, 46]), array([24, 17, 21, 45, 61, 71, 47, 31, 14, 26]))

    """
    if isinstance(x, list):
        x = np.array(x)
    assert type(x) == np.ndarray

    kernel = np.ones(1 + (2 * guard_len) + (2 * noise_len), dtype=x.dtype) / (2 * noise_len)
    kernel[noise_len:noise_len + (2 * guard_len) + 1] = 0

    noise_floor = convolve1d(x, kernel, mode=mode)
    threshold = noise_floor + l_bound

    return threshold, noise_floor


def caso(x, *argv, **kwargs):
    """Detects peaks in signal using Cell-Averaging Smallest-Of CFAR (CASO-CFAR).

    Args:
        x (~numpy.ndarray): Signal.
        *argv: See mmwave.dsp.cfar.caso\_
        **kwargs: See mmwave.dsp.cfar.caso\_

    Returns:
        ~numpy.ndarray: Boolean array of detected peaks in x.

    Examples:
        >>> signal = np.random.randint(100, size=10)
        >>> signal
            array([41, 76, 95, 28, 25, 53, 10, 93, 54, 85])
        >>> det = mm.dsp.caso(signal, l_bound=20, guard_len=1, noise_len=3)
        >>> det
            array([False, False,  True, False, False, False, False,  True,  True,
                    True])

        Perform a non-wrapping CFAR

        >>> signal = np.random.randint(100, size=10)
        >>> signal
            array([41, 76, 95, 28, 25, 53, 10, 93, 54, 85])
        >>> det =  mm.dsp.caso(signal, l_bound=20, guard_len=1, noise_len=3, mode='constant')
        >>> det
            array([False,  True,  True, False, False, False, False,  True,  True,
                    True])

    """
    if isinstance(x, list):
        x = np.array(x)
    threshold, _ = caso_(x, *argv, **kwargs)
    ret = (x > threshold)
    return ret


def caso_(x, guard_len=4, noise_len=8, mode='wrap', l_bound=4000):
    """Uses Cell-Averaging Smallest-Of CFAR (CASO-CFAR) to calculate a threshold that can be used to calculate peaks in a signal.

    Args:
        x (~numpy.ndarray): Signal.
        guard_len (int): Number of samples adjacent to the CUT that are ignored.
        noise_len (int): Number of samples adjacent to the guard padding that are factored into the calculation.
        mode (str): Specify how to deal with edge cells.
        l_bound (float or int): Additive lower bound while calculating peak threshold.

    Returns:
        Tuple [ndarray, ndarray]
            1. (ndarray): Upper bound of noise threshold.
            #. (ndarray): Raw noise strength.

    Examples:
        >>> signal = np.random.randint(100, size=10)
        >>> signal
            array([41, 76, 95, 28, 25, 53, 10, 93, 54, 85])
        >>> threshold = mm.dsp.caso_(signal, l_bound=20, guard_len=1, noise_len=3)
        >>> (threshold[0].astype(int), threshold[1].astype(int))
            (array([69, 55, 49, 72, 72, 86, 69, 55, 49, 72]), array([49, 35, 29, 52, 52, 66, 49, 35, 29, 52]))

        Perform a non-wrapping CFAR thresholding

        >>> signal = np.random.randint(100, size=10)
        >>> signal
            array([41, 76, 95, 28, 25, 53, 10, 93, 54, 85])
        >>> threshold = mm.dsp.caso_(signal, l_bound=20, guard_len=1, noise_len=3, mode='constant')
        >>> (threshold[0].astype(int), threshold[1].astype(int))
            (array([69, 55, 49, 72, 72, 86, 69, 55, 49, 72]), array([49, 35, 29, 52, 52, 66, 49, 35, 29, 52]))

    """
    if isinstance(x, list):
        x = np.array(x)

    l_window, r_window = _cfar_windows(x, guard_len, noise_len, mode)

    # Generate scaling based on mode
    l_window = l_window / noise_len
    r_window = r_window / noise_len
    if mode == 'wrap':
        noise_floor = np.minimum(l_window, r_window)
    elif mode == 'constant':
        edge_cells = guard_len + noise_len
        noise_floor = np.minimum(l_window, r_window)
        noise_floor[:edge_cells] = r_window[:edge_cells]
        noise_floor[-edge_cells:] = l_window[-edge_cells:]
    else:
        raise ValueError(f'Mode {mode} is not a supported mode')

    threshold = noise_floor + l_bound
    return threshold, noise_floor


def cago(x, *argv, **kwargs):
    """Detects peaks in signal using Cell-Averaging Greatest-Of CFAR (CAGO-CFAR).

    Args:
        x (~numpy.ndarray): Signal.
        *argv: See mmwave.dsp.cfar.cago\_
        **kwargs: See mmwave.dsp.cfar.cago\_

    Returns:
        ~numpy.ndarray: Boolean array of detected peaks in x.

    Examples:
        >>> signal = np.random.randint(100, size=10)
        >>> signal
            array([41, 76, 95, 28, 25, 53, 10, 93, 54, 85])
        >>> det = mm.dsp.cago(signal, l_bound=20, guard_len=1, noise_len=3)
        >>> det
            array([False, False,  True, False, False, False, False,  True, False,
                    False])

        Perform a non-wrapping CFAR

        >>> signal = np.random.randint(100, size=10)
        >>> signal
            array([41, 76, 95, 28, 25, 53, 10, 93, 54, 85])
        >>> det = mm.dsp.cago(signal, l_bound=20, guard_len=1, noise_len=3, mode='constant')
        >>> det
            array([False,  True,  True, False, False, False, False,  True,  True,
                    True])

    """
    if isinstance(x, list):
        x = np.array(x)
    threshold, _ = cago_(x, *argv, **kwargs)
    ret = (x > threshold)
    return ret


def cago_(x, guard_len=4, noise_len=8, mode='wrap', l_bound=4000):
    """Uses Cell-Averaging Greatest-Of CFAR (CAGO-CFAR) to calculate a threshold that can be used to calculate peaks in a signal.

    Args:
        x (~numpy.ndarray): Signal.
        guard_len (int): Number of samples adjacent to the CUT that are ignored.
        noise_len (int): Number of samples adjacent to the guard padding that are factored into the calculation.
        mode (str): Specify how to deal with edge cells.
        l_bound (float or int): Additive lower bound while calculating peak threshold.

    Returns:
        Tuple [ndarray, ndarray]
            1. (ndarray): Upper bound of noise threshold.
            #. (ndarray): Raw noise strength.

    Examples:
        >>> signal = np.random.randint(100, size=10)
        >>> signal
            array([41, 76, 95, 28, 25, 53, 10, 93, 54, 85])
        >>> threshold = mm.dsp.cago_(signal, l_bound=20, guard_len=1, noise_len=3)
        >>> (threshold[0].astype(int), threshold[1].astype(int))
            (array([72, 97, 80, 87, 90, 97, 80, 87, 90, 86]), array([52, 77, 60, 67, 70, 77, 60, 67, 70, 66]))

        Perform a non-wrapping CFAR thresholding

        >>> signal = np.random.randint(100, size=10)
        >>> signal
            array([41, 76, 95, 28, 25, 53, 10, 93, 54, 85])
        >>> threshold = mm.dsp.cago_(signal, l_bound=20, guard_len=1, noise_len=3, mode='constant')
        >>> (threshold[0].astype(int), threshold[1].astype(int))
            (array([69, 55, 49, 72, 90, 97, 69, 55, 49, 72]), array([49, 35, 29, 52, 70, 77, 49, 35, 29, 52]))

    """
    if isinstance(x, list):
        x = np.array(x)

    l_window, r_window = _cfar_windows(x, guard_len, noise_len, mode)

    # Generate scaling based on mode
    l_window = l_window / noise_len
    r_window = r_window / noise_len
    if mode == 'wrap':
        noise_floor = np.maximum(l_window, r_window)
    elif mode == 'constant':
        edge_cells = guard_len + noise_len
        noise_floor = np.maximum(l_window, r_window)
        noise_floor[:edge_cells] = r_window[:edge_cells]
        noise_floor[-edge_cells:] = l_window[-edge_cells:]
    else:
        raise ValueError(f'Mode {mode} is not a supported mode')

    threshold = noise_floor + l_bound
    return threshold, noise_floor


def os(x, *argv, **kwargs):
    """Performs Ordered-Statistic CFAR (OS-CFAR) detection on the input array.

    Args:
        x (~numpy.ndarray): Noisy array to perform cfar on with log values
        *argv: See mmwave.dsp.cfar.os\_
        **kwargs: See mmwave.dsp.cfar.os\_


    Returns:
        ~numpy.ndarray: Boolean array of detected peaks in x.

    Examples:
        >>> signal = np.random.randint(100, size=10)
        >>> signal
            array([41, 76, 95, 28, 25, 53, 10, 93, 54, 85])
        >>> det = mm.dsp.os(signal, k=3, scale=1.1, guard_len=0, noise_len=3)
        >>> det
            array([False,  True,  True, False, False, False, False,  True, False,
                    True])

    """
    if isinstance(x, list):
        x = np.array(x)
    threshold, _ = os_(x, *argv, **kwargs)
    ret = (x > threshold)
    return ret


def os_(x, guard_len=0, noise_len=8, k=12, scale=1.0):
    """Performs Ordered-Statistic CFAR (OS-CFAR) detection on the input array.

    Args:
        x (~numpy.ndarray): Noisy array to perform cfar on with log values
        guard_len (int): Number of samples adjacent to the CUT that are ignored.
        noise_len (int): Number of samples adjacent to the guard padding that are factored into the calculation.
        k (int): Ordered statistic rank to sample from.
        scale (float): Scaling factor.

    Returns:
        Tuple [ndarray, ndarray]
            1. (ndarray): Upper bound of noise threshold.
            #. (ndarray): Raw noise strength.

    Examples:
        >>> signal = np.random.randint(100, size=10)
        >>> signal
            array([41, 76, 95, 28, 25, 53, 10, 93, 54, 85])
        >>> threshold = mm.dsp.os_(signal, k=3, scale=1.1, guard_len=0, noise_len=3)
        >>> (threshold[0].astype(int), threshold[1].astype(int))
            (array([93, 59, 58, 58, 83, 59, 59, 58, 83, 83]), array([85, 54, 53, 53, 76, 54, 54, 53, 76, 76]))

    """
    if isinstance(x, list):
        x = np.array(x, dtype=np.uint32)

    n = len(x)
    noise_floor = np.zeros(n)
    threshold = np.zeros(n, dtype=np.float32)
    cut_idx = -1

    # Initial CUT
    left_idx = list(np.arange(n - noise_len - guard_len - 1, n - guard_len - 1))
    right_idx = list(np.arange(guard_len, guard_len + noise_len))

    # All other CUTs
    while cut_idx < (n - 1):
        cut_idx += 1

        left_idx.pop(0)
        left_idx.append((cut_idx - 1) % n)

        right_idx.pop(0)
        right_idx.append((cut_idx + guard_len + noise_len) % n)

        window = np.concatenate((x[left_idx], x[right_idx]))
        window.partition(k)
        noise_floor[cut_idx] = window[k]
        threshold[cut_idx] = noise_floor[cut_idx] * scale

    return threshold, noise_floor


def _cfar_windows(x, guard_len, noise_len, mode):
    if type(x) != np.ndarray:
        raise TypeError(f'Expected array-like input got {type(x)}')

    # Create kernels
    r_kernel = np.zeros(1 + (2 * guard_len) + (2 * noise_len), dtype=x.dtype)
    r_kernel[:noise_len] = 1
    l_kernel = r_kernel[::-1]

    # Do initial convolutions
    l_window = convolve1d(x, l_kernel, mode=mode)
    r_window = convolve1d(x, r_kernel, mode=mode)

    return l_window, r_window


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
