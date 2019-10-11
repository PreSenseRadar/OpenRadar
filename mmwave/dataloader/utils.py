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


def parse_tsw1400(path, num_chirps_per_frame, num_frames, num_ants, num_adc_samples, iq=True, num_adc_bits=16):
    """Parse the raw ADC data based on xWR16xx/IWR6843 and TSW1400 configuration.

    Parse the row-majored binary output from raw ADC data capture to numpy ndarray with the shape
    of (numFrame, num_chirps_per_frame, num_ants, num_adc_samples). For more details, refer to the original document  
    https://www.ti.com/lit/an/swra581b/swra581b.pdf.
    
    Args:
        path (str): File path of the binary data.
        num_chirps_per_frame (int): Total number of chirps from all transmitters in a single frame.
        num_frames (int): Number of frames in the recorded binary data.
        num_ants (int): Number of physical receivers.
        num_adc_samples (int): Number of ADC samples.
        iq (bool): True if complex and False if real.
        num_adc_bits (int): Number of ADC quantization bits.
    
    Returns:
        ndarray: Parsed ADC data with the shape of (num_frames, num_chirps_per_frame, num_ants, num_adc_samples)
    
    Example:
        >>> # Suppose your binary data is located at "./data/radar_data.bin".
        >>> adc_data = parse_tsw1400("./data/radar_data.bin", 128, 200, 4, 256)
        >>> # Now your adc_data will be an ndarray with shape (200, 128, 4, 256) and dtype as complex.
    """
    channel_count = iq + 1  # always 2 in this case
    num_chirps = num_chirps_per_frame * num_frames
    adc_row = num_chirps * num_ants
    adc_col = channel_count * num_adc_samples
    num_sample = adc_row * adc_col

    adc_data = np.fromfile(path, dtype=np.uint16)
    assert adc_data.shape[0] == num_sample, \
        "Actual number of samples (%d) doesn\'t equal to expected (%d)" % (adc_data.shape[0], num_sample)

    # Raw data is in "offset binary format", so need to subtract 2**15 in order to get two's-complement.
    offset = np.array([2 ** 15], dtype=np.int16)
    adc_data = np.subtract(adc_data, offset, dtype=np.int16)

    if num_adc_bits != 16:
        l_max = 2 ** (16 - 1) - 1
        idx_threshold = adc_data > l_max
        adc_data[idx_threshold] -= 2 ** 16

    adc_data = adc_data.reshape((num_chirps, num_ants, adc_col))

    if iq:
        adc_deinterleaved = [adc_data[:, :, i::channel_count] for i in range(channel_count)]  # i = 0, 1, channel_count = 2
        adc_data = adc_deinterleaved[0] + 1j * adc_deinterleaved[1]

    # adc_data *= normFactor
    assert adc_data.shape == (num_chirps, num_ants, num_adc_samples), \
        "ADC data is not parsed to desired shape. Currently it is {}".format(adc_data.shape)

    adc_data = adc_data.reshape(num_frames, num_chirps_per_frame, num_ants, num_adc_samples)

    return adc_data
