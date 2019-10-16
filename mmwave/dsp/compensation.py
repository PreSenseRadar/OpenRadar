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

import math
from .utils import *


def _generate_dft_sin_cos_table(dft_length):
    """Generate SIN/COS table for doppler compensation reference.

    Generate SIN/COS table. Also generates Sine/Cosine at half, one thrid and two thirds the bin value. This is a helper
    function only called by add_doppler_compensation().

    Args:
        dft_len: (int) dft_len Length of the DFT. It is used as numDopperBins, which is numCHirpsPerFrame/numTxAntenns.

    Returns:
        dft_sin_cos_table (np.ndarray): ndarray in complex format with generated sine (image) cosine (real) table.
        bins (np.ndarray): Sin/Cos at half, one thrid and two thirds the bin.
    """
    dft_sin_cos_table = np.arange(dft_length, dtype=np.float32)
    dft_sin_cos_table = np.cos(2 * np.pi * dft_sin_cos_table / dft_length) + \
                        1j * -np.sin(2 * np.pi * dft_sin_cos_table / dft_length)

    # 1/2, 1/3 and 2/3 bins
    bins = np.array([0.5, 1.0/3, 2.0/3])
    bins = np.cos(2 * np.pi * bins / dft_length) - 1j * np.sin(2 * np.pi * bins / dft_length)

    return dft_sin_cos_table, bins

def add_doppler_compensation(input_data,
                             num_tx_antennas,
                             doppler_indices=None,
                             num_doppler_bins=None):
    """Compensation of Doppler phase shift in the virtual antennas.

    Compensation of Doppler phase shift on the virtual antennas (corresponding to second or third Tx antenna chirps). 
    Symbols corresponding to virtual antennas, are rotated by half of the Doppler phase shift measured by Doppler FFT 
    for 2 Tx system and 1/3 and 2/3 of the Doppler phase shift for 3 Tx system. The phase shift read from the table 
    using half or 1/3 of the object Doppler index value. If the Doppler index is odd, an extra half of the bin phase 
    shift is added.

    The original function is called per detected objects. This functions is modified to directly compensate the 
    azimuth_in matrix (numDetObj, num_angle_bins)

    Args:
        input_data (ndarray): (range, num_antennas, doppler) Radar data cube that needs to be compensated. It can be the input
            of azimuth FFT after CFAR or the intermediate right before beamforming.
        num_tx_antennas (int): Number of transmitters.
        num_doppler_bins (int): (Optional) Number of doppler bins in the radar data cube. If given, that means the doppler
            indices are signed and needs to be converted to unsigned.
        doppler_indices (ndarray): (Optional) Doppler index of the object with the shape of (num_detected_objects). If given, 
            that means we only compensate on selected doppler bins.
    
    Return:
        input_data (ndarray): Original input data with the columns related to virtual receivers got compensated.
        
    Example:
        >>> # If the compensation is done right before naive azimuth FFT and objects is detected already. you need to 
        >>> # feed in the doppler_indices
        >>> dataIn = add_doppler_compensation(dataIn, 3, doppler_indices, 128)
  """
    num_antennas = input_data.shape[1]
    if num_tx_antennas == 1:
        return input_data
    elif num_tx_antennas > 3:
        raise ValueError("the specified number of transimitters is currently not supported")

    # Call the gen function above to generate the tables.
    azimuth_mod_coefs, bins = _generate_dft_sin_cos_table(int(num_doppler_bins))
    
    # Convert signed doppler indices to unsigned and divide Doppler index by 2.
    if doppler_indices is not None:
        if num_doppler_bins is not None:
            doppler_compensation_indices = doppler_indices & (num_doppler_bins - 1)
            doppler_compensation_indices[doppler_compensation_indices[:] >= (num_doppler_bins / 2)] -= num_doppler_bins
            doppler_compensation_indices = doppler_compensation_indices // 2
            doppler_compensation_indices[doppler_compensation_indices[:] < 0] += num_doppler_bins
        exp_doppler_compensation = azimuth_mod_coefs[doppler_compensation_indices]
    else:
        exp_doppler_compensation = azimuth_mod_coefs

    # Add half bin rotation if Doppler index was odd
    if num_tx_antennas == 2:
        exp_doppler_compensation[(doppler_indices[:] % 2) == 1] *= bins[0]
    else:
        exp_doppler_compensation[(doppler_indices[:] % 3) == 1] *= bins[1]
        exp_doppler_compensation[(doppler_indices[:] % 3) == 2] *= bins[2]

    # Expand the dim so that the broadcasting below will work.
    exp_doppler_compensation = np.expand_dims(exp_doppler_compensation, axis=1)

    # Rotate
    azimuth_values = input_data[:, (num_antennas/num_tx_antennas):, :]
    for azi_val in azimuth_values:
        Re = exp_doppler_compensation.real * azi_val.imag - exp_doppler_compensation.imag * azi_val.real
        Im = exp_doppler_compensation.imag * azi_val.imag + exp_doppler_compensation.real * azi_val.real
        input_data[:, (num_antennas/num_tx_antennas):, :] = Re + 1j * Im

    return input_data


def rx_channel_phase_bias_compensation(rx_channel_compensations, input, num_antennas):
    """Compensation of rx channel phase bias.

    Args:
        rx_channel_compensations: rx channel compensation coefficient.
        input: complex number.
        num_antennas: number of symbols.
    """
    azimuth_values = input[:num_antennas]
    rx_channel_compensations_values = rx_channel_compensations[:num_antennas]

    Re = rx_channel_compensations_values * (azimuth_values.imag - azimuth_values.real)
    Im = rx_channel_compensations_values * (azimuth_values.imag + azimuth_values.real)
    input[:num_antennas] = Re + 1j * Im

    return


def near_field_correction(idx,
                          detected_objects,
                          start_range_index, 
                          end_range_index,
                          azimuth_input,
                          azimuth_output,
                          num_angle_bins, 
                          num_rx_antennas, 
                          range_resolution):
    """Correct phase error as the far-field plane wave assumption breaks.

    Calculates near field correction for input detected index (corresponding
    to a range position). Referring to top level doxygen @ref 
    nearFieldImplementation, this function performs the Set 1 rotation with the 
    correction and adds to Set 0 in place to produce result in Set 0 of the 
    azimuth_output.

    This correction is done per detected objects from CFAR detection

    Args:
        idx: index of the detected objects in detected_objects.
        detected_objects: detected objects matrix with dimension of 100 x 6, where it holds at most 100 objects and 6 members are 
            rangeIdx, dopplerIdx, peakVal, x, y and z. It is configured as a structured array.
        start_range_index: start range index of near field correction.
        end_range_index: end range index of near field correction.
        azimuth_input: complex array of which length is num_angle_bins+numVirtualAntAzim, where numVirtualAntAzim = 4, 8 or 12 
            depending on how many Txs are used.
    
    Returns:
        None. azimuth_output is changed in-place.
    """

    LAMBDA_77GHz_MILLIMETER = 3e8 / 77e9
    MMWDEMO_TWO_PI_OVER_LAMBDA = 2.0 * math.pi / LAMBDA_77GHz_MILLIMETER

    # Sanity check and check if nearFieldCorrection is necessary.
    assert idx >= 0 and idx < MAX_OBJ_OUT, "idx is out of bound!"
    rangeIdx = detected_objects['rangeIdx'][idx]
    if rangeIdx < start_range_index or rangeIdx >= end_range_index:
        print("{} is out of the nearFieldCorrection range".format(rangeIdx))
        return

    # num_angle_bins = 64
    azimuth_input[:num_angle_bins] = 0
    azimuth_input[num_rx_antennas: num_rx_antennas + num_rx_antennas] = azimuth_input[num_angle_bins:]

    # azimuth_output has length of 2*num_angle_bins.
    azimuth_output[num_angle_bins:] = np.fft.fft(azimuth_input, n=num_angle_bins)

    # #define MMWDEMO_NEAR_FIELD_A (0)
    # B can be changed to position the desired reference (boresight) in the geometry */
    # #define MMWDEMO_NEAR_FIELD_B (LAMBDA_77GHz_MILLIMETER) //((((2 + 0.75) * LAMBDA_77GHz_MILLIMETER)) + 8.7)
    # #define MMWDEMO_NEAR_FIELD_C (2 * LAMBDA_77GHz_MILLIMETER)
    # 8.7 mm is the actual (approximate) measurement of distance between tx1 and rx4,
    # measured using a wooden scale that has a resolution of 1 mm
    # #define MMWDEMO_NEAR_FIELD_D (MMWDEMO_NEAR_FIELD_C + 8.7)
    # #define MMWDEMO_NEAR_FIELD_E (MMWDEMO_NEAR_FIELD_D + 1.5 * LAMBDA_77GHz_MILLIMETER)
    geometry_points = {"A": 0,
                       "B": LAMBDA_77GHz_MILLIMETER,
                       "C": 2 * LAMBDA_77GHz_MILLIMETER,
                       "D": 2 * LAMBDA_77GHz_MILLIMETER + 8.7,
                       "E": (2 + 1.5) * LAMBDA_77GHz_MILLIMETER + 8.7}

    # AB, CB, DB, EB
    geometry_lines = np.array([geometry_points["A"] - geometry_points["B"],
                              geometry_points["C"] - geometry_points["B"],
                              geometry_points["D"] - geometry_points["B"],
                              geometry_points["E"] - geometry_points["B"]])

    geometry_lines_square = geometry_lines * geometry_lines

    range_in_millimeter = (detected_objects['rangeIdx'][idx] * range_resolution - range_resolution) * 1000
    range_squared = range_in_millimeter * range_in_millimeter
    theta_incrementation = 2.0 / num_angle_bins

    for i in range(num_angle_bins):
        theta = i * theta_incrementation if i < num_angle_bins / 2 else (i - num_angle_bins) * theta_incrementation

        tx1 = np.sqrt(range_squared + geometry_lines_square[1] - range_in_millimeter * theta * geometry_lines[1] * 2)
        rx4 = np.sqrt(range_squared + geometry_lines_square[2] - range_in_millimeter * theta * geometry_lines[2] * 2)
        tx2 = np.sqrt(range_squared + geometry_lines_square[0] - range_in_millimeter * theta * geometry_lines[0] * 2)
        rx1 = np.sqrt(range_squared + geometry_lines_square[3] - range_in_millimeter * theta * geometry_lines[3] * 2)

        if range > 0:
            psi = MMWDEMO_TWO_PI_OVER_LAMBDA * ((tx2 + rx1) - (rx4 + tx1)) - np.pi * theta
            corrReal = np.cos(psi)
            corrImag = np.sin(-psi)

        out1CorrReal = azimuth_output[num_angle_bins + i].real * corrReal + \
                       azimuth_output[num_angle_bins + i].imag * corrImag
        out1CorrImag = azimuth_output[num_angle_bins + i].imag * corrReal + \
                       azimuth_output[num_angle_bins + i].real * corrImag

        azimuth_output[i] = (azimuth_output[i].real + out1CorrReal) + \
                        (azimuth_output[i].imag + out1CorrImag) * 1j

    return


def dc_range_signature_removal(fft_out1_d,
                               positive_bin_idx,
                               negative_bin_idx,
                               calib_dc_range_sig_cfg,
                               num_tx_antennas,
                               num_chirps_per_frame):
    """Compensation of DC range antenna signature.

    Antenna coupling signature dominates the range bins close to the radar. These are the bins in the range FFT output 
    located around DC. This feature is under user control in terms of enable/disable and start/end range bins through a 
    CLI command called calibDcRangeSig. During measurement (when the CLI command is issued with feature enabled), each 
    of the specified range bins for each of the virtual antennas are accumulated over the specified number of chirps 
    and at the end of the period, the average is computed for each bin/antenna combination for removal after the 
    measurement period is over. Note that the number of chirps to average must be power of 2. It is assumed that no 
    objects are present in the vicinity of the radar during this measurement period. After measurement is done, the 
    removal starts for all subsequent frames during which each of the bin/antenna average estimate is subtracted from 
    the corresponding received samples in real-time for subsequent processing.

    This function has a measurement phase while calib_dc_range_sig_cfg.counter is less than the preferred value and calibration 
    phase afterwards. The original function is performed per chirp. Here it is modified to be called per frame.

    Args:
        fft_out1_d: (num_chirps_per_frame, num_rx_antennas, numRangeBins). Output of 1D FFT.
        positive_bin_idx: the first positive_bin_idx range bins (inclusive) to be compensated.
        negative_bin_idx: the last -negative_bin_idx range bins to be compensated.
        calib_dc_range_sig_cfg: a simple class for calibration configuration's storing purpose.
        num_tx_antennas: number of transmitters.
        num_chirps_per_frame: number of total chirps per frame.
      
    Returns:
        None. fft_out1_d is modified in-place.
    """
    if not calib_dc_range_sig_cfg.counter:
        calib_dc_range_sig_cfg.mean.fill(0)

    # Calibration
    if calib_dc_range_sig_cfg.counter < calib_dc_range_sig_cfg.num_frames * num_tx_antennas:
        # Accumulate
        calib_dc_range_sig_cfg.mean[0, :positive_bin_idx + 1] = np.sum(
            fft_out1_d[0::2, :, :positive_bin_idx + 1],
            axis=(0, 1))
        calib_dc_range_sig_cfg.mean[0, positive_bin_idx + 1:] = np.sum(fft_out1_d[0::2, :, negative_bin_idx:],
                                                                                 axis=(0, 1))

        calib_dc_range_sig_cfg.mean[1, :positive_bin_idx + 1] = np.sum(
            fft_out1_d[1::2, :, :positive_bin_idx + 1],
            axis=(0, 1))
        calib_dc_range_sig_cfg.mean[1, positive_bin_idx + 1:] = np.sum(fft_out1_d[1::2, :, negative_bin_idx:],
                                                                                 axis=(0, 1))

        calib_dc_range_sig_cfg.counter += 1

        if calib_dc_range_sig_cfg.counter == (calib_dc_range_sig_cfg.num_frames * num_tx_antennas):
            # Divide
            num_avg_chirps = calib_dc_range_sig_cfg.num_frames * num_chirps_per_frame
            calib_dc_range_sig_cfg.mean /= num_avg_chirps

    else:
        # fft_out1_d -= mean
        fft_out1_d[0::2, :, :positive_bin_idx + 1] -= calib_dc_range_sig_cfg.mean[0, :positive_bin_idx + 1]
        fft_out1_d[0::2, :, positive_bin_idx + 1:] -= calib_dc_range_sig_cfg.mean[0, positive_bin_idx + 1:]
        fft_out1_d[1::2, :, :positive_bin_idx + 1] -= calib_dc_range_sig_cfg.mean[1, :positive_bin_idx + 1]
        fft_out1_d[1::2, :, positive_bin_idx + 1:] -= calib_dc_range_sig_cfg.mean[1, positive_bin_idx + 1:]


def clutter_removal(input_val, axis=0):
    """Perform basic static clutter removal by removing the mean from the input_val on the specified doppler axis.

    Args:
        input_val (ndarray): Array to perform static clutter removal on. Usually applied before performing doppler FFT.
            e.g. [num_chirps, num_vx_antennas, num_samples], it is applied along the first axis.
        axis (int): Axis to calculate mean of pre-doppler.

    Returns:
        ndarray: Array with static clutter removed.

    """
    # Reorder the axes
    reordering = np.arange(len(input_val.shape))
    reordering[0] = axis
    reordering[axis] = 0
    input_val = input_val.transpose(reordering)

    # Apply static clutter removal
    mean = input_val.transpose(reordering).mean(0)
    output_val = input_val - mean

    return output_val.transpose(reordering)
