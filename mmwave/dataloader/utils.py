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


def parse_tsw1400(path, num_chirps_per_frame, num_frames, num_rx_ant, num_adc_samples, iq=True, num_adc_bits=16):
    """Parse the raw ADC data based on IWR1642 and TSW1400 configuration.

    Parse the row-majored binary output from raw ADC data capture to numpy array with the shape
    of (numFrame, num_chirps_per_frame, num_rxs, num_adc_samples)
    
    Args:
        path (str): File path of the binary data.
        num_chirps_per_frame: Total number of chirps from all transmitters in a single frame.
        num_frames: Number of frames in the recorded binary data.
        num_rx_ant: Number of physical receivers.
        num_adc_samples: Number of ADC samples.
        iq: True if complex and False if real.
        num_adc_bits: Number of ADC quantization bits.
    
    Returns:
        adc (ndarray): Parsed ADC data with the shape of (num_frames, num_chirps_per_frame, num_rxs, num_adc_samples)
    """
    channel_count = iq + 1  # always 2 in this case
    num_chirps = num_chirps_per_frame * num_frames
    adc_row = num_chirps * num_rx_ant
    adc_col = channel_count * num_adc_samples
    num_sample = adc_row * adc_col

    adc = np.fromfile(path, dtype=np.uint16)
    assert adc.shape[0] == num_sample, \
        "Actual number of samples (%d) doesn\'t equal to expected (%d)" % (adc.shape[0], num_sample)

    offset = np.array([2 ** 15], dtype=np.int16)
    adc = np.subtract(adc, offset, dtype=np.int16)

    if num_adc_bits != 16:
        l_max = 2 ** (16 - 1) - 1
        idx_threshold = adc > l_max
        adc[idx_threshold] -= 2 ^ 16

    adc = adc.reshape((num_chirps, num_rx_ant, adc_col))

    if iq:
        adc_deinterleaved = [adc[:, :, i::channel_count] for i in range(channel_count)]  # i = 0, 1, channel_count = 2
        adc = adc_deinterleaved[0] + 1j * adc_deinterleaved[1]
        assert adc.dtype == np.complex_ or adc.dtype == np.complex64, \
            "ADC data should be complex, currently it is {}".format(adc.dtype)

    # adc *= normFactor
    assert adc.shape == (num_chirps, num_rx_ant, num_adc_samples), \
        "ADC data is not parsed to desired shape. Currently it is {}".format(adc.shape)

    adc = adc.reshape(num_frames, num_chirps_per_frame, num_rx_ant, num_adc_samples)

    return adc
