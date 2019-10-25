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
from .utils import *
from . import compensation
from scipy.signal import find_peaks
import warnings

def azimuth_processing(radar_cube, det_obj_2d, config, window_type_2d=None):
    """Calculate the X/Y coordinates for all detected objects.
    
    The following procedures will be performed in this function:

    1. Filter radarCube based on the range indices from detObj2D and optional clutter removal.
    2. Re-do windowing and 2D FFT, select associated doppler indices to form the azimuth input.
    3. Doppler compensation on the virtual antennas related to tx2. Save optional copy for near field compensation and\
      velocity disambiguation.
    4. Perform azimuth FFT.
    #. Optional near field correction and velocity disambiguation. Currently mutual exclusive.
    #. Magnitude squared.
    #. Calculate X/Y coordinates.
    
    Args:
        radar_cube: (numChirpsPerFrame, numRxAntennas, numRangeBins). Because of the space limitation, TI Demo starts again
            from the 1D FFT, recalculate the 2D FFT on the selected range bins and pick the associated doppler bins for the 
            azimuth FFT. 
        det_obj_2d: (numDetObj, 3)
        config: [TBD]
        window_type_2d: windowing function for the 2D FFT. Default is None.
    
    Returns:
        azimuthOut: (numDetObj, 5). Copy of detObj2D but with populated X/Y coordinates.
    """
    # detObj2D: [numDetObj, 3], where the columns are rangeIdx, dopplerIdx and peakVal
    # 1. Filter 1D-FFT output along range direction and perform optional static clutter removal.
    # fft2d_azimuth_in: [numChirpsPerFrame, numRxAntennas, numDetObj]
    num_det_obj = det_obj_2d.shape[0]
    fft2d_azimuth_in = radar_cube[..., det_obj_2d[:, RANGEIDX].astype(np.uint32)]

    # Clutter Removal
    if config.clutterRemovalEnabled:
        fft2d_azimuth_in = compensation.clutterRemoval(fft2d_azimuth_in)

    # Rearrange just like what is done for 1st 2D FFT calculation
    # BE CAREFUL OF THE NAMING!
    fft2d_azimuth_in = np.concatenate((fft2d_azimuth_in[0::2, ...], fft2d_azimuth_in[1::2, ...]), axis=1)
    # transpose to (numDetObj, numVirtualAntennas, numDopplerBins)
    fft2d_azimuth_in = np.transpose(fft2d_azimuth_in, axes=(2, 1, 0))

    # 2. Perform windowing, filter along doppler direction and then full-FFT or single-point DFT.
    # Windowing 32x32
    fft2d_window_type = None
    if fft2d_window_type:
        fft2d_azimuth_in = windowing(fft2d_azimuth_in, fft2d_window_type, axis=2)
    else:
        fft2d_azimuth_in = fft2d_azimuth_in
    # FFT 32x32
    # Will worry about single-point DFT later.
    fft2d_azimuth_out = np.fft.fft(fft2d_azimuth_in)
    fft2d_azimuth_out = np.fft.fftshift(fft2d_azimuth_out, axes=2)
    # Filter fft2d_azimuth_out with the DopplerIdx from CFAR and PG
    # azimuth_in(numDetObj, numVirtualAntennas)
    azimuth_in = np.zeros((num_det_obj, config.numAngleBins), dtype=np.complex_)
    azimuth_in[:, :config.numVirtualAntAzim] = np.array([fft2d_azimuth_out[i, :, dopplerIdx] for i, dopplerIdx in
                                                         enumerate(
                                                             det_obj_2d[:, DOPPLERIDX].astype(np.uint32))]).squeeze()

    assert azimuth_in.shape == (num_det_obj, config.numAngleBins), \
        "azimuth FFT input dimension is wrong, it should be {} instead of {}." \
            .format((num_det_obj, config.numAngleBins), azimuth_in.shape)

    # 3. Doppler compensation.
    # Only the 2nd half of the 2D FFT output needs to be compensated. That is 
    # the reason for azimuth_in[:, numRxAntennas, :].
    compensation.addDopplerCompensation(det_obj_2d[:, DOPPLERIDX],
                                        compensation.azimuthModCoefs,
                                        compensation.azimuthModCoefsHalfBin,
                                        azimuth_in[:, config.numRxAntennas:],
                                        config.numRxAntennas * (config.numTxAntennas - 1))

    # If receiver channel biases are provided, activate the following function
    # to compensate for it.
    if config.rxChannelComp is not None:
        compensation.rxChanPhaseBiasCompensation(config.rxChannelComp,
                                                 azimuth_in,
                                                 config.numVirtualAntennas)

    # The near field correction is currently in exclusion with velocity 
    # disambiguation because of implementation complexities and also because it 
    # is unlikely to have objects at high velocities in the near field.

    # Save a copy for flipped version of azimuth_in for velocity disambiguation.
    azimuth_in_copy = np.zeros_like(azimuth_in)
    if config.extendedMaxVelocityEnabled and config.nearFieldCorrectionCfg.enabled:
        assert False, "Extended maximum velocity and near field correction are not supported simultaneously."

    if config.extendedMaxVelocityEnabled:
        azimuth_in_copy[:, :config.numVirtualAntAzim] = azimuth_in[:, :config.numVirtualAntAzim]

    # Save a copy of RX antennas corresponding to Tx2 antenna.
    if config.nearFieldCorrectionCfg.enabled:
        idx_temp = ((det_obj_2d[:, RANGEIDX] >= config.nearFieldCorrectionCfg.startRangeIdx) &
                    (det_obj_2d[:, RANGEIDX] <= config.nearFieldCorrectionCfg.endRangeIdx))
        azimuth_in_copy[idx_temp, :config.numRxAntennas] = \
            azimuth_in_copy[idx_temp, config.numRxAntennas:config.numRxAntennas * 2]
        azimuth_in[idx_temp, config.numRxAntennas:config.numRxAntennas * 2] = 0

    # 4. 3rd FFT.
    azimuth_out = np.fft.fft(azimuth_in)

    # 5.1. Optional near field correction.
    if config.nearFieldCorrectionCfg.enabled:
        azimuth_out_copy = np.fft.fft(azimuth_in_copy)
        compensation.nearFieldCorrection(det_obj_2d,
                                         config.nearFieldCorrectionCfg.startRangeIdx,
                                         config.nearFieldCorrectionCfg.endRangeIdx,
                                         azimuth_in, azimuth_in_copy,
                                         azimuth_out, azimuth_out_copy)

    # 5.2. Optional velocity disambiguation.
    if config.extendedMaxVelocityEnabled:
        azimuth_in = azimuth_in_copy
        azimuth_in[:, config.numRxAntennas:config.numVirtualAntAzim] *= -1

    # 6. Magnitude squared.
    azimuth_mag_sqr = np.abs(azimuth_out) ** 2

    # 7. Azimuth, X/Y calculation and populate detObj2D. Convert doppler index to 
    det_obj2d_azimuth = compensation.XYestimation(azimuth_mag_sqr, config.numAngleBins, det_obj_2d)
    det_obj2d_azimuth[:, DOPPLERIDX] = DOPPLER_IDX_TO_SIGNED(det_obj2d_azimuth[:, DOPPLERIDX], config.numDopplerBins)

    return det_obj2d_azimuth


# -- PreSense built, Python optimized AOA functions --


"""PreSense built, Python optimized AOA functions

Description:
    Angle of Arrival Implementations for 1D ULA arrays along with useful helper functions

Angle of Arrival Functions:
    aoa_bartlett     -- AOA function using Bartlett Method   (Beamforming Based)
    aoa_capon        -- AOA function using Capon Method      (Beamforming Based)

Helper Functions:
    forward_backward_avg  --     Calculate a forward-backward averaged covariance matrix
    cov_matrix           --     Calculate the covariance of the input_data (Rxx)

Constants:
    rx      = number of of antennas in the array
    chirps  = number of samples (chirps) per frame
    P       = number of thetas in the estimated angle spectrum
"""


# ------------------------------- PreSense Beamforming Functions -------------------------------

def aoa_bartlett(steering_vec, sig_in, axis):
    """Perform AOA estimation using Bartlett Beamforming on a given input signal (sig_in). Make sure to specify the correct axis in (axis)
    to ensure correct matrix multiplication. The power spectrum is calculated using the following equation:

    .. math::
        P_{ca} (\\theta) = a^{H}(\\theta) R_{xx}^{-1} a(\\theta)

    This steers the beam using the steering vector as weights:

    .. math::
        w_{ca} (\\theta) = a(\\theta)

    Args:
        steering_vec (ndarray): A 2D-array of size (numTheta, num_ant) generated from gen_steering_vec
        sig_in (ndarray): Either a 2D-array or 3D-array of size (num_ant, numChirps) or (numChirps, num_vrx, num_adc_samples) respectively, containing ADC sample data sliced as described
        axis (int): Specifies the axis where the Vrx data in contained.

    Returns:
        doa_spectrum (ndarray): A 3D-array of size (numChirps, numThetas, numSamples)

    Example:
        >>> # In this example, dataIn is the input data organized as numFrames by RDC
        >>> frame = 0
        >>> dataIn = np.random.rand((num_frames, num_chirps, num_vrx, num_adc_samples))
        >>> aoa_bartlett(steering_vec,dataIn[frame],axis=1)
    """
    y = np.matmul(np.conjugate(steering_vec), sig_in.swapaxes(axis, np.arange(len(sig_in.shape))[-2]))
    doa_spectrum = np.abs(y) ** 2
    return doa_spectrum.swapaxes(axis, np.arange(len(sig_in.shape))[-2])


def aoa_capon(x, steering_vector, magnitude=False):
    """Perform AOA estimation using Capon (MVDR) Beamforming on a rx by chirp slice

    Calculate the aoa spectrum via capon beamforming method using one full frame as input.
    This should be performed for each range bin to achieve AOA estimation for a full frame
    This function will calculate both the angle spectrum and corresponding Capon weights using
    the equations prescribed below.

    .. math::
        P_{ca} (\\theta) = \\frac{1}{a^{H}(\\theta) R_{xx}^{-1} a(\\theta)}
        
        w_{ca} (\\theta) = \\frac{R_{xx}^{-1} a(\\theta)}{a^{H}(\\theta) R_{xx}^{-1} a(\\theta)}

    Args:
        x (ndarray): Output of the 1d range fft with shape (num_ant, numChirps)
        steering_vector (ndarray): A 2D-array of size (numTheta, num_ant) generated from gen_steering_vec
        magnitude (bool): Azimuth theta bins should return complex data (False) or magnitude data (True). Default=False

    Raises:
        ValueError: steering_vector and or x are not the correct shape

    Returns:
        A list containing numVec and steeringVectors
        den (ndarray: A 1D-Array of size (numTheta) containing azimuth angle estimations for the given range
        weights (ndarray): A 1D-Array of size (num_ant) containing the Capon weights for the given input data
    
    Example:
        >>> # In this example, dataIn is the input data organized as numFrames by RDC
        >>> Frame = 0
        >>> dataIn = np.random.rand((num_frames, num_chirps, num_vrx, num_adc_samples))
        >>> for i in range(256):
        >>>     scan_aoa_capon[i,:], _ = dss.aoa_capon(dataIn[Frame,:,:,i].T, steering_vector, magnitude=True)

    """

    if steering_vector.shape[1] != x.shape[0]:
        raise ValueError("'steering_vector' with shape (%d,%d) cannot matrix multiply 'input_data' with shape (%d,%d)" \
        % (steering_vector.shape[0], steering_vector.shape[1], x.shape[0], x.shape[1]))

    Rxx = cov_matrix(x)
    Rxx = forward_backward_avg(Rxx)
    Rxx_inv = np.linalg.inv(Rxx)
    # Calculate Covariance Matrix Rxx
    first = Rxx_inv @ steering_vector.T
    den = np.reciprocal(np.einsum('ij,ij->i', steering_vector.conj(), first.T))
    weights = np.matmul(first, den)

    if magnitude:
        return np.abs(den), weights
    else:
        return den, weights


# ------------------------------- HELPER FUNCTIONS -------------------------------

def cov_matrix(x):
    """ Calculates the spatial covariance matrix (Rxx) for a given set of input data (x=inputData). 
        Assumes rows denote Vrx axis.

    Args:
        x (ndarray): A 2D-Array with shape (rx, adc_samples) slice of the output of the 1D range fft

    Returns:
        Rxx (ndarray): A 2D-Array with shape (rx, rx)
    """
    
    if x.ndim > 2:
        raise ValueError("x has more than 2 dimensions.")

    if x.shape[0] > x.shape[1]:
        warnings.warn("cov_matrix input should have Vrx as rows. Needs to be transposed", RuntimeWarning)
        x = x.T

    _, num_adc_samples = x.shape
    Rxx = x @ np.conjugate(x.T)
    Rxx = np.divide(Rxx, num_adc_samples)

    return Rxx


def forward_backward_avg(Rxx):
    """ Performs forward backward averaging on the given input square matrix

    Args:
        Rxx (ndarray): A 2D-Array square matrix containing the covariance matrix for the given input data

    Returns:
        R_fb (ndarray): The 2D-Array square matrix containing the forward backward averaged covariance matrix
    """
    assert np.size(Rxx, 0) == np.size(Rxx, 1)

    # --> Calculation
    M = np.size(Rxx, 0)  # Find number of antenna elements
    Rxx = np.matrix(Rxx)  # Cast np.ndarray as a np.matrix

    # Create exchange matrix
    J = np.eye(M)  # Generates an identity matrix with row/col size M
    J = np.fliplr(J)  # Flips the identity matrix left right
    J = np.matrix(J)  # Cast np.ndarray as a np.matrix

    R_fb = 0.5 * (Rxx + J * np.conjugate(Rxx) * J)

    return np.array(R_fb)


def peak_search(doa_spectrum, peak_threshold_weight=0.251188643150958):
    """ Wrapper function to perform scipy.signal's prescribed peak search algorithm
        Tested Runtime: 45 µs ± 2.61 µs per loop (mean ± std. dev. of 7 runs, 10000 loops each)

    Args:
        doa_spectrum (ndarray): A 1D-Array of size (numTheta, 1) containing the theta spectrum at a given range bin
        peak_threshold_weight (float): A float specifying the desired peak declaration threshold weight to be applied

    Returns:
        num_max (int): The number of max points found by the algorithm
        peaks (list): List of indexes where peaks are located
        total_power (float): Total power in the current spectrum slice

    """

    peak_threshold = max(doa_spectrum) * peak_threshold_weight
    peaks, properties = find_peaks(doa_spectrum, height=peak_threshold)
    num_max = len(peaks)
    total_power = np.sum(properties['peak_heights'])
    return num_max, peaks, total_power


def peak_search_full(doa_spectrum, gamma=1.2, peak_threshold_weight=0.251188643150958):
    """ Perform TI prescribed peak search algorithm
    Tested Runtime: 147 µs ± 4.27 µs per loop (mean ± std. dev. of 7 runs, 10000 loops each)

    Args:
        doa_spectrum (ndarray): A 1D-Array of size (numTheta, 1) containing the theta spectrum at a given range bin
        gamma (float): A float specifying the maximum/minimum wiggle necessary to qualify as a peak
        peak_threshold_weight (float): A float specifying the desired peak declaration threshold weight to be applied

    Returns:
        num_max (int): The number of max points found by the algorithm
        ang_est (list): List of indexes where the peaks are located

    """

    ang_est = np.zeros(4, dtype='int')
    # Prevent thinking a sidelobe is a peak
    peak_threshold = max(doa_spectrum) * peak_threshold_weight
    # Perform Multiple Peak Search
    steering_vec_size = len(doa_spectrum)
    running_idx = 0
    num_max = 0
    extend_loc = 0
    init_stage = True
    max_val = 0
    min_val = np.inf
    max_loc = 0
    locate_max = False

    while running_idx < (steering_vec_size + extend_loc):
        if running_idx >= steering_vec_size:
            local_index = running_idx - steering_vec_size
        else:
            local_index = running_idx

        current_val = doa_spectrum[local_index]

        # Record local maximum value and its location
        if current_val > max_val:
            max_val = current_val
            max_loc = local_index

        # Record local minimum value and its location
        if current_val < min_val:
            min_val = current_val

        if locate_max:
            # Perform peak search
            if current_val < max_val / gamma:
                # Curve has begun dipping. Prev max val is a Local max. Check if it is a sidelobe
                if max_val >= peak_threshold:
                    # Curve has dipped and it is not a sidelobe. Target found.
                    ang_est[num_max] = max_loc
                    num_max += 1
                min_val = current_val
                locate_max = False
        else:
            if current_val > min_val * gamma:
                locate_max = True
                max_val = current_val
                if init_stage:
                    extend_loc = running_idx
                    init_stage = False
        running_idx += 1

    return num_max, ang_est


def peak_search_full_variance(doa_spectrum, steering_vec_size, sidelobe_level=0.251188643150958, gamma=1.2):
    """ Performs peak search (TI's full search) will retaining details about each peak including
    each peak's width, location, and value.

    Args:
        doa_spectrum (ndarray): a 1D numpy array containing the power spectrum generated via some aoa method (naive,
        bartlett, or capon)
        steering_vec_size (int): Size of the steering vector in terms of number of theta bins
        sidelobe_level (float): A low value threshold used to avoid sidelobe detections as peaks
        gamma (float): Weight to determine when a peak will pass as a true peak

    Returns:
        peak_data (ndarray): A 1D numpy array of custom data types with length numberOfPeaksDetected.
        Each detected peak is organized as [peak_location, peak_value, peak_width]
        total_power (float): The total power of the spectrum. Used for variance calculations
    """
    peak_threshold = max(doa_spectrum) * sidelobe_level

    # Multiple Peak Search
    running_index = 0
    num_max = 0
    extend_loc = 0
    init_stage = True
    max_val = 0
    total_power = 0
    max_loc = 0
    max_loc_r = 0
    min_val = np.inf
    locate_max = False

    peak_data = []

    while running_index < (steering_vec_size + extend_loc):
        if running_index >= steering_vec_size:
            local_index = running_index - steering_vec_size
        else:
            local_index = running_index

        # Pull local_index values
        current_val = doa_spectrum[local_index]
        # Record Min & Max locations
        if current_val > max_val:
            max_val = current_val
            max_loc = local_index
            max_loc_r = running_index

        if current_val < min_val:
            min_val = current_val

        if locate_max:
            if current_val < max_val / gamma:
                if max_val > peak_threshold:
                    bandwidth = running_index - max_loc_r
                    obj = dict.fromkeys(['peakLoc', 'peakVal', 'peakWid'])
                    obj['peakLoc'] = max_loc
                    obj['peakVal'] = max_val
                    obj['peakWid'] = bandwidth
                    peak_data.append(obj)
                    total_power += max_val
                    num_max += 1
                min_val = current_val
                locate_max = False
        else:
            if current_val > min_val * gamma:
                locate_max = True
                max_val = current_val
                if init_stage:
                    extend_loc = running_index
                    init_stage = False

        running_index += 1

    peak_data = np.array(peak_data)
    return peak_data, total_power


def variance_estimation(num_max, est_resolution, peak_data, total_power, width_adjust_3d_b=2.5, input_snr=10000):
    """ This function will calculate an estimated variance value for each detected peak. This should
        be run after running peak_search_full_variance

    Args:
        num_max (int): The number of detected peaks
        est_resolution (float): The desired resolution in terms of theta
        peak_data (ndarray): A numpy array of dictionaries, where each dictionary is of the form: {"peakLoc": , "peakVal": , "peakWid": }
        total_power (float): The total power of the spectrum
        width_adjust_3d_b (float): Constant to adjust the gamma bandwidth to 3dB level
        input_snr (int): the linear snr for the input signal samples

    Returns:
        est_var (ndarray): A 1D array of variances (of the peaks). The order of the peaks is preserved from peak_data
    """
    est_var = np.zeros(num_max)
    for objIndex in range(num_max):
        peak_width = 2 * est_resolution * peak_data[objIndex]['peakWid'] * width_adjust_3d_b
        snr = 2 * input_snr * peak_data[objIndex]['peakVal'] / total_power

        temp_interpol = np.sqrt(np.reciprocal(snr))  # sqrt(1/snr)

        est_var[objIndex] = (peak_width * temp_interpol)
    return est_var


def gen_steering_vec(ang_est_range, ang_est_resolution, num_ant):
    """Generate a steering vector for AOA estimation given the theta range, theta resolution, and number of antennas

    Defines a method for generating steering vector data input --Python optimized Matrix format
    The generated steering vector will span from -angEstRange to angEstRange with increments of ang_est_resolution
    The generated steering vector should be used for all further AOA estimations (bartlett/capon)

    Args:
        ang_est_range (int): The desired span of thetas for the angle spectrum.
        ang_est_resolution (float): The desired resolution in terms of theta
        num_ant (int): The number of Vrx antenna signals captured in the RDC

    Returns:
        num_vec (int): Number of vectors generated (integer divide angEstRange/ang_est_resolution)
        steering_vectors (ndarray): The generated 2D-array steering vector of size (num_vec,num_ant)

    Example:
        >>> #This will generate a numpy array containing the steering vector with 
        >>> #angular span from -90 to 90 in increments of 1 degree for a 4 Vrx platform
        >>> _, steering_vec = gen_steering_vec(90,1,4)

    """
    num_vec = (2 * ang_est_range / ang_est_resolution + 1)
    num_vec = int(round(num_vec))
    steering_vectors = np.zeros((num_vec, num_ant), dtype='complex64')
    for kk in range(num_vec):
        for jj in range(num_ant):
            mag = -1 * np.pi * jj * np.sin((-ang_est_range + kk * ang_est_resolution) * np.pi / 180)
            real = np.cos(mag)
            imag = np.sin(mag)

            steering_vectors[kk, jj] = np.complex(real, imag)

    return [num_vec, steering_vectors]


# ------------------------------- TI BEAMFORMING FUNCTIONS -------------------------------
def aoa_estimation_bf_one_point(num_ant, sig_in, steering_vec):
    """ Calculates the total power of the given spectrum

    Args:
        num_ant (int): The number of virtual antennas (Vrx) being used
        sig_in (ndarray): A 2D-array of size (num_ant, numChirps) containing ADC sample data sliced for used Vrx
        steering_vec (ndarray): A 2D-array of size (numTheta, num_ant) generated from gen_steering_vec

    Returns:
        out_value (complex): The total power of the given input spectrum
    """

    # --- UNOPTIMIZED CODE ---
    #    for idx in range(steering_vec_size):
    #        f2temp1 = sig_in[0]+(np.conjugate(steering_vec[((num_ant-1)*idx)])*sig_in[1])
    #        for i in range(2, num_ant):
    #            f2temp1 = f2temp1+(np.conjugate(steering_vec[((num_ant-1)*idx)+(i-1)])*sig_in[i])
    # --- UNOPTIMIZED CODE ---

    assert sig_in.shape[0] == num_ant, "[ERROR] Shape of sig_in does not meet required num_ant dimensions"
    out_value = np.matmul(steering_vec[:num_ant], sig_in)
    return out_value


# Single Peak Angle of Arrival Estimation -- Detection Only
def aoa_est_bf_single_peak_det(sig_in, steering_vec):
    """Beamforming Estimate Angle of Arrival for single peak (single peak should be known a priori)
        Function call does not include variance calculations
        Function does not generate a spectrum. Rather, it only returns the array index (theta) to the highest peak

    Args:
        sig_in (ndarray): A 2D-array of size (num_ant, numChirps) containing ADC sample data sliced as described
        steering_vec (ndarray): A generated 2D-array steering vector of size (numVec,num_ant)

    Returns:
        max_index (int): Index of the theta spectrum at a given range bin that contains the max peak
    """

    # OPTIMIZED
    y = np.matmul(np.conjugate(steering_vec), sig_in)
    doa_spectrum = np.abs(y) ** 2
    max_index = np.argmax(doa_spectrum)

    return max_index


# Single Peak Angle of Arrival Estimation -- Variance, Spectrum, AOA 
def aoa_est_bf_single_peak(num_ant, noise, est_resolution, sig_in, steering_vec_size, steering_vec):
    """Beamforming Estimate Angle of Arrival for single peak (single peak should be known a priori)
        Function call includes variance calculations
        Function does generate a spectrum.

    Args:
        num_ant (int): The number of virtual receivers in the current radar setup
        noise (float): Input noise figure
        est_resolution (float): Desired theta spectrum resolution used when generating steering_vec
        sig_in (ndarray): A 2D-array of size (num_ant, numChirps) containing ADC sample data sliced as described
        steering_vec_size (int): Length of the steering vector array
        steering_vec (ndarray): A generated 2D-array steering vector of size (numVec,num_ant)

    Returns:
        est_var (float): The estimated variance of the doa_spectrum
        max_index (int): Index of the theta spectrum at a given range bin that contains the max peak
        doa_spectrum (ndarray): A 1D-Array of size (numTheta, 1) containing the theta spectrum at a given range bin
    """
    aoaestbf_var_est_const = 1
    input_power = 0

    # Calculate input_power
    for i in range(num_ant):
        input_power += np.real(sig_in[i]) ** 2 + np.imag(sig_in[i]) ** 2

    # OPTIMIZED
    y = np.matmul(np.conjugate(steering_vec), sig_in)
    doa_spectrum = np.abs(y) ** 2
    max_index = np.argmax(doa_spectrum)
    max_power = doa_spectrum[max_index]
    total_power = doa_spectrum.sum()

    # Begin mainlobe bandwidth calculation
    threshold_3db = max_power * 0.5
    signal_power = 0
    left_index = max_index
    right_index = max_index + 1

    # Find mainlobe 3dB left threshold point
    while doa_spectrum[left_index] >= threshold_3db and left_index >= 0:
        signal_power += doa_spectrum[left_index]
        left_index -= 1
        if left_index < 0:
            left_index = steering_vec_size - 1

    # Find mainlobe 3dB right threshold point
    while right_index < steering_vec_size and doa_spectrum[right_index] >= threshold_3db:
        signal_power += doa_spectrum[right_index]
        right_index += 1
        if right_index == steering_vec_size:
            right_index = 0

    temp_3db_span = right_index - (left_index + 1)
    # If Right and Left indexes are not on right and left side of mainlobe, add offset
    if temp_3db_span < 0:
        temp_3db_span += steering_vec_size

    temp_var_sqr_inv = 2 * (aoaestbf_var_est_const ** 2) * input_power * num_ant * signal_power
    temp_var_sqr_inv *= np.reciprocal(noise * total_power)
    temp_interpol = np.sqrt(np.reciprocal(temp_var_sqr_inv))

    return [est_resolution * temp_3db_span * temp_interpol, max_index,
            doa_spectrum]  # Return [est_var, angleEst, angleSpectrum]


# Multiple Peak Angle of Arrival Estimation -- Detection Only
def aoa_est_bf_multi_peak_det(gamma, sidelobe_level, sig_in, steering_vec, steering_vec_size, ang_est, search=False):
    """Use Bartlett beamforming to estimate AOA for multi peak situation (a priori), no variance calculation

    Args:
        gamma (float): Weight to determine when a peak will pass as a true peak
        sidelobe_level (float): A low value threshold used to avoid sidelobe detections as peaks
        sig_in (ndarray): A 2D-array of size (num_ant, numChirps) containing ADC sample data sliced as described
        steering_vec (ndarray): A generated 2D-array steering vector of size (numVec,num_ant)
        steering_vec_size (int): Length of the steering vector array
        ang_est (ndarray): An empty 1D numpy array that gets populated with max indexes
        search (bool): Flag that determines whether search is done to find max points

    Returns:
        num_max (int): The number of max points found across the theta bins at this particular range bin
        doa_spectrum (ndarray): A 1D-Array of size (numTheta, 1) containing the theta spectrum at a given range bin
        
    """

    # Calculate Total Power at each angle

    # -- OPTIMIZED -- Vectorized version
    y = np.matmul(np.conjugate(steering_vec), sig_in)
    doa_spectrum = np.abs(y) ** 2
    max_pow = np.max(doa_spectrum)
    # -- OPTIMIZED --

    if search:
        #        # Prevent thinking a sidelobe is a peak
        peak_threshold = max_pow * sidelobe_level
        #        print("Threshold: ", peak_threshold)
        # Perform Multiple Peak Search
        running_idx = 0
        num_max = 0
        extend_loc = 0
        init_stage = True
        max_val = 0
        min_val = np.inf
        max_loc = 0
        locate_max = False

        while running_idx < (steering_vec_size + extend_loc):
            if running_idx >= steering_vec_size:
                local_index = running_idx - steering_vec_size
            else:
                local_index = running_idx

            current_val = doa_spectrum[local_index]

            # Record local maximum value and its location
            if current_val > max_val:
                max_val = current_val
                max_loc = local_index

            # Record local minimum value and its location
            if current_val < min_val:
                min_val = current_val

            if locate_max:
                # Perform peak search
                if current_val < max_val / gamma:
                    # Curve has begun dipping. Prev max val is a Local max. Check if it is a sidelobe
                    if max_val >= peak_threshold:
                        # Curve has dipped and it is not a sidelobe. Target found.
                        ang_est[num_max] = max_loc
                        num_max += 1
                    min_val = current_val
                    locate_max = False
            else:
                if current_val > min_val * gamma:
                    locate_max = True
                    max_val = current_val
                    if init_stage:
                        extend_loc = running_idx
                        init_stage = False
            running_idx += 1

    else:
        num_max = -1
    return num_max, doa_spectrum


# Multiple Peak Angle of Arrival Estimation -- Full Variance Calculations
def aoa_est_bf_multi_peak(gamma, sidelobe_level, width_adjust_3d_b, input_snr, est_resolution, sig_in, steering_vec,
                          steering_vec_size, peak_data, ang_est):
    """ This function performs all sections of the angle of arrival process in one function.
    
    1. Performs bartlett beamforming
    2. Performs multi-peak search
    3. Calculates an estimated variance
    
    Args:
        gamma (float): Weight to determine when a peak will pass as a true peak
        sidelobe_level (float): A low value threshold used to avoid sidelobe detections as peaks
        width_adjust_3d_b (float): Constant to adjust gamma bandwidth to 3dB bandwidth
        input_snr (float): Input data SNR value
        est_resolution (float): User defined target resolution
        sig_in (ndarray): A 2D-array of size (num_ant, numChirps) containing ADC sample data sliced as described
        steering_vec (ndarray): A generated 2D-array steering vector of size (numVec,num_ant)
        steering_vec_size (int): Length of the steering vector array
        peak_data (ndarray): A 2D ndarray with custom data-type that contains information on each detected point
        ang_est (ndarray): An empty 1D numpy array that gets populated with max indexes

    Returns:
        Tuple [ndarray, ndarray]
            1. num_max (int): The number of max values detected by search algorithm
            #. est_var (ndarray): The estimated variance of this range of thetas at this range bin
    """

    est_var = []
    aoaestbf_var_est_const = 1  # Standard value, serves no purpose

    # -- OPTIMIZED --
    y = np.matmul(np.conjugate(steering_vec), sig_in)
    doa_spectrum = np.abs(y) ** 2
    max_index = np.argmax(doa_spectrum)
    max_power = doa_spectrum[max_index]
    peak_threshold = max_power * sidelobe_level
    # -- OPTIMIZED --

    # Multiple Peak Search
    running_index = 0
    num_max = 0
    extend_loc = 0
    init_stage = True
    max_val = 0
    total_power = 0
    max_loc = 0
    max_loc_r = 0
    min_val = np.inf
    locate_max = False

    while running_index < (steering_vec_size + extend_loc):
        if running_index >= steering_vec_size:
            local_index = running_index - steering_vec_size
        else:
            local_index = running_index

        # Pull local_index values
        current_val = doa_spectrum[local_index]
        # Record Min & Max locations
        if current_val > max_val:
            max_val = current_val
            max_loc = local_index
            max_loc_r = running_index

        if current_val < min_val:
            min_val = current_val

        if locate_max:
            if current_val < max_val / gamma:
                if max_val > peak_threshold:
                    bandwidth = running_index - max_loc_r
                    obj = peak_data[num_max]
                    obj['peakLoc'] = max_loc
                    obj['peakVal'] = max_val
                    obj['peakWid'] = bandwidth
                    total_power += max_val
                    num_max += 1
                min_val = current_val
                locate_max = False
        else:
            if current_val > min_val * gamma:
                locate_max = True
                max_val = current_val
                if init_stage:
                    extend_loc = running_index
                    init_stage = False

        running_index += 1

    # Variance Estimation
    for objIndex in range(num_max):
        peak_width = 2 * est_resolution * peak_data[objIndex]['peakWid'] * width_adjust_3d_b
        snr = 2 * input_snr * peak_data[objIndex]['peakVal'] / total_power

        temp_interpol = np.sqrt(np.reciprocal(snr))  # sqrt(1/snr)

        est_var.append((peak_width / aoaestbf_var_est_const) * temp_interpol)
        ang_est[objIndex] = peak_data[objIndex]['peakLoc']

    return num_max, np.array(est_var)


def naive_xyz(virtual_ant, num_tx=3, num_rx=4, fft_size=64):
    """ Estimate the phase introduced from the elevation of the elevation antennas

    Args:
        virtual_ant: Signal received by the rx antennas, shape = [#angleBins, #detectedObjs], zero-pad #virtualAnts to #angleBins
        num_tx: Number of transmitter antennas used
        num_rx: Number of receiver antennas used
        fft_size: Size of the fft performed on the signals

    Returns:
        x_vector (float): Estimated x axis coordinate in meters (m)
        y_vector (float): Estimated y axis coordinate in meters (m)
        z_vector (float): Estimated z axis coordinate in meters (m)

    """
    assert num_tx > 2, "need a config for more than 2 TXs"
    num_detected_obj = virtual_ant.shape[1]

    # Zero pad azimuth
    azimuth_ant = virtual_ant[:2 * num_rx, :]
    azimuth_ant_padded = np.zeros(shape=(fft_size, num_detected_obj), dtype=np.complex_)
    azimuth_ant_padded[:2 * num_rx, :] = azimuth_ant

    # Process azimuth information
    azimuth_fft = np.fft.fft(azimuth_ant_padded, axis=0)
    k_max = np.argmax(np.abs(azimuth_fft), axis=0)  # shape = (num_detected_obj, )
    # peak_1 = azimuth_fft[k_max]
    peak_1 = np.zeros_like(k_max, dtype=np.complex_)
    for i in range(len(k_max)):
        peak_1[i] = azimuth_fft[k_max[i], i]

    k_max[k_max > (fft_size // 2) - 1] = k_max[k_max > (fft_size // 2) - 1] - fft_size
    wx = 2 * np.pi / fft_size * k_max  # shape = (num_detected_obj, )
    x_vector = wx / np.pi

    # Zero pad elevation
    elevation_ant = virtual_ant[2 * num_rx:, :]
    elevation_ant_padded = np.zeros(shape=(fft_size, num_detected_obj), dtype=np.complex_)
    # elevation_ant_padded[:len(elevation_ant)] = elevation_ant
    elevation_ant_padded[:num_rx, :] = elevation_ant

    # Process elevation information
    elevation_fft = np.fft.fft(elevation_ant, axis=0)
    elevation_max = np.argmax(np.log2(np.abs(elevation_fft)), axis=0)  # shape = (num_detected_obj, )
    peak_2 = np.zeros_like(elevation_max, dtype=np.complex_)
    # peak_2 = elevation_fft[np.argmax(np.log2(np.abs(elevation_fft)))]
    for i in range(len(elevation_max)):
        peak_2[i] = elevation_fft[elevation_max[i], i]

    # Calculate elevation phase shift
    wz = np.angle(peak_1 * peak_2.conj() * np.exp(1j * 2 * wx))
    z_vector = wz / np.pi
    y_vector = np.sqrt(1 - x_vector ** 2 - z_vector ** 2)
    return x_vector, y_vector, z_vector


def beamforming_naive_mixed_xyz(azimuth_input, input_ranges, range_resolution, method='Capon', num_vrx=12, est_range=90,
                                est_resolution=1):
    """ This function estimates the XYZ location of a series of input detections by performing beamforming on the
    azimuth axis and naive AOA on the vertical axis.
        
    TI xWR1843 virtual antenna map
    Row 1               8  9  10 11
    Row 2         0  1  2  3  4  5  6  7

    phi (ndarray):
    theta (ndarray):
    ranges (ndarray):
    xyz_vec (ndarray):

    Args:
        azimuth_input (ndarray): Must be a numpy array of shape (numDetections, numVrx)
        input_ranges (ndarray): Numpy array containing the rangeBins that have detections (will determine x, y, z for
        each detection)
        range_resolution (float): The range_resolution in meters per rangeBin for rangeBin->meter conversion
        method (string): Determines which beamforming method to use for azimuth aoa estimation.
        num_vrx (int): Number of virtual antennas in the radar platform. Default set to 12 for 1843
        est_range (int): The desired span of thetas for the angle spectrum. Used for gen_steering_vec
        est_resolution (float): The desired angular resolution for gen_steering_vec

    Raises:
        ValueError: If method is not one of two AOA implementations ('Capon', 'Bartlett')
        ValueError: azimuthInput's second axis should have same shape as the number of Vrx

    Returns:
        tuple [ndarray, ndarray, ndarray, ndarray, list]:
            1. A numpy array of shape (numDetections, ) where each element represents the elevation angle in degrees
            #. A numpy array of shape (numDetections, ) where each element represents the azimuth in degrees
            #. A numpy array of shape (numDetections, ) where each element represents the polar range in rangeBins
            #. A numpy array of shape (3, numDetections) and format: [x, y, z] where x, y, z are 1D arrays. x, y, z \
            should be in meters

    """
    if method not in ('Capon', 'Bartlett'):
        raise ValueError("Method argument must be 'Capon' or 'Bartlett'")

    if azimuth_input.shape[1] != num_vrx:
        raise ValueError("azimuthInput is the wrong shape. Change num_vrx if not using TI 1843 platform")

    doa_var_thr = 10
    num_vec, steering_vec = gen_steering_vec(est_range, est_resolution, 8)

    output_e_angles = []
    output_a_angles = []
    output_ranges = []

    for i, inputSignal in enumerate(azimuth_input):
        if method == 'Capon':
            doa_spectrum, _ = aoa_capon(np.reshape(inputSignal[:8], (8, 1)).T, steering_vec)
            doa_spectrum = np.abs(doa_spectrum)
        elif method == 'Bartlett':
            doa_spectrum = aoa_bartlett(steering_vec, np.reshape(inputSignal[:8], (8, 1)), axis=0)
            doa_spectrum = np.abs(doa_spectrum).squeeze()
        else:
            doa_spectrum = None

        # Find Max Values and Max Indices

        #    num_out, max_theta, total_power = peak_search(doa_spectrum)
        obj_dict, total_power = peak_search_full_variance(doa_spectrum, steering_vec.shape[0], sidelobe_level=0.9)
        num_out = len(obj_dict)
        max_theta = [obj['peakLoc'] for obj in obj_dict]

        estimated_variance = variance_estimation(num_out, est_resolution, obj_dict, total_power)

        higher_rung = inputSignal[8:12]
        lower_rung = inputSignal[2:6]
        for j in range(num_out):
            ele_out = aoa_estimation_bf_one_point(4, higher_rung, steering_vec[max_theta[j]])
            azi_out = aoa_estimation_bf_one_point(4, lower_rung, steering_vec[max_theta[j]])
            num = azi_out * np.conj(ele_out)
            wz = np.arctan2(num.imag, num.real) / np.pi

            temp_angle = -est_range + max_theta[
                j] * est_resolution  # Converts to degrees, centered at boresight (0 degrees)
            # Make sure the temp angle generated is within bounds
            if np.abs(temp_angle) <= est_range and estimated_variance[j] < doa_var_thr:
                e_angle = np.arcsin(wz)
                a_angle = -1 * (np.pi / 180) * temp_angle  # Degrees to radians
                output_e_angles.append((180 / np.pi) * e_angle)  # Convert radians to degrees

                # print(e_angle)
                # if (np.sin(a_angle)/np.cos(e_angle)) > 1 or (np.sin(a_angle)/np.cos(e_angle)) < -1:
                # print("Found you", (np.sin(a_angle)/np.cos(e_angle)))
                # assert np.cos(e_angle) == np.nan, "Found you"

                # TODO: Not sure how to deal with arg of arcsin >1 or <-1
#                if np.sin(a_angle)/np.cos(e_angle) > 1:
#                    output_a_angles.append((180 / np.pi) * np.arcsin(1))
#                    print("Found a pesky nan")
#                elif np.sin(a_angle)/np.cos(e_angle) < -1:
#                    output_a_angles.append((180 / np.pi) * np.arcsin(-1))
#                    print("Found a pesky nan")
#                else:
#                    output_a_angles.append((180 / np.pi) * np.arcsin(np.sin(a_angle)/np.cos(e_angle))) # Why

                output_a_angles.append((180 / np.pi) * np.arcsin(np.sin(a_angle) * np.cos(e_angle)))  # Why

                output_ranges.append(input_ranges[i])

    phi = np.array(output_e_angles)
    theta = np.array(output_a_angles)
    ranges = np.array(output_ranges)

    # points could be calculated by trigonometry,
    x = np.sin(np.pi / 180 * theta) * ranges * range_resolution     # x = np.sin(azi) * range
    y = np.cos(np.pi / 180 * theta) * ranges * range_resolution     # y = np.cos(azi) * range
    z = np.tan(np.pi / 180 * phi) * ranges * range_resolution       # z = np.tan(ele) * range

    xyz_vec = np.array([x, y, z])

    # return phi, theta, ranges
    return phi, theta, ranges, xyz_vec
