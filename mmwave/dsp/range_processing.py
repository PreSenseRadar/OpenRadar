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
from . import utils


def range_resolution(num_adc_samples, dig_out_sample_rate=2500, freq_slope_const=60.012):
    """ Calculate the range resolution for the given radar configuration

    Args:
        num_adc_samples (int): The number of given ADC samples in a chirp
        dig_out_sample_rate (int): The ADC sample rate
        freq_slope_const (float): The slope of the freq increase in each chirp

    Returns:
        tuple [float, float]:
            range_resolution (float): The range resolution for this bin
            band_width (float): The bandwidth of the radar chirp config
    """
    light_speed_meter_per_sec = 299792458
    freq_slope_m_hz_per_usec = freq_slope_const
    adc_sample_period_usec = 1000.0 / dig_out_sample_rate * num_adc_samples
    band_width = freq_slope_m_hz_per_usec * adc_sample_period_usec * 1e6
    range_resolution = light_speed_meter_per_sec / (2.0 * band_width)

    return range_resolution, band_width


def range_processing(adc_data, window_type_1d=None, axis=-1):
    """Perform 1D FFT on complex-format ADC data.

    Perform optional windowing and 1D FFT on the ADC data.

    Args:
        adc_data (ndarray): (num_chirps_per_frame, num_rx_antennas, num_adc_samples). Performed on each frame. adc_data
                            is in complex by default. Complex is float32/float32 by default.
        window_type_1d (mmwave.dsp.utils.Window): Optional window type on 1D FFT input. Default is None. Can be selected
                                                from Bartlett, Blackman, Hanning and Hamming.
    
    Returns:
        radar_cube (ndarray): (num_chirps_per_frame, num_rx_antennas, num_range_bins). Also called fft_1d_out
    """
    # windowing numA x numB suggests the coefficients is numA-bits while the 
    # input and output are numB-bits. Same rule applies to the FFT.
    fft1d_window_type = window_type_1d
    if fft1d_window_type:
        fft1d_in = utils.windowing(adc_data, fft1d_window_type, axis=axis)
    else:
        fft1d_in = adc_data

    # Note: np.fft.fft is a 1D operation, using higher dimension input defaults to slicing last axis for transformation
    radar_cube = np.fft.fft(fft1d_in, axis=axis)

    return radar_cube


def zoom_range_processing(adc_data, low_freq, high_freq, fs, d, resample_number):
    """Perform ZoomFFT on complex-format ADC data in a user-defined frequency range.

    Args:
        adc_data (ndarray): (num_chirps_per_frame, num_rx_antennas, num_adc_samples). Performed on each frame. adc_data
                            is in complex by default. Complex is float32/float32 by default.
        low_freq (int): a user-defined number which specifies the lower bound on the range of frequency spectrum which
                        the user would like to zoom on
        high_freq (int): a user-defined number which specifies the higher bound on the range of frequency spectrum which
                         the user would like to zoom on
        fs (int) : sampling rate of the original signal
        d (int): Sample spacing (inverse of the sampling rate)
        resample_number (int): The number of samples in the re-sampled signal.
    
    Returns:
        zoom_fft_spectrum (ndarray): (num_chirps_per_frame, num_rx_antennas, resample_number).
    """
    # adc_data shape: [num_chirps_per_frame, num_rx_antennas, num_range_bins]
    num_chirps_per_frame = adc_data.shape[0]
    num_rx_antennas = adc_data.shape[1]
    # num_range_bins = adc_data.shape[2]

    zoom_fft_spectrum = np.zeros(shape=(num_chirps_per_frame, num_rx_antennas, resample_number))

    for i in range(num_chirps_per_frame):
        for j in range(num_rx_antennas):
            zoom_fft_inst = ZoomFFT.ZoomFFT(low_freq, high_freq, fs, adc_data[i, j, :])
            zoom_fft_inst.compute_fft()
            zoom_fft_spectrum[i, j, :] = zoom_fft_inst.compute_zoomfft()

    return zoom_fft_spectrum


def zoom_fft_visualize(zoom_fft_spectrum, antenna_idx, range_bin_idx):
    '''to be implemented'''
    pass
