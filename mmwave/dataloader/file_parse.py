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
from mmwave.dataloader import DCA1000
import os
import struct


def parse_raw_adc(source_fp, dest_fp):
    """Reads a binary data file containing raw adc data from a DCA1000, cleans it and saves it for manual processing.

    Note:
        "Raw adc data" in this context refers to the fact that the DCA1000 initially sends packets of data containing
        meta data and is merged with actual pure adc data. Part of the purpose of this function is to remove this
        meta data.

    Note:
        TODO: Support zero fill missing packets
        TODO: Support reordering packets

    Args:
        source_fp (str): Path to raw binary adc data.
        dest_fp (str): Path to output cleaned binary adc data.

    Returns:
        None

    """
    buff = np.fromfile(source_fp, dtype=np.uint8)
    packets_recv = 0
    buff_pos = 0
    adc_data = []
    while buff_pos < len(buff):
        packets_recv += 1

        # Index binary data
        sequence_info = buff[buff_pos:buff_pos + 4]
        length_info = buff[buff_pos + 4:buff_pos + 8]
        # bytes_info = buff[buff_pos + 8:buff_pos + 14]
        buff_pos += 14

        # Unpack binary data
        packet_num = struct.unpack('<1l', sequence_info)[0]
        packet_length = struct.unpack('<l', length_info.tobytes())[0]
        # curr_bytes_read = struct.unpack('<Q', np.pad(bytes_info, (0, 2), mode='constant').tobytes())[0]

        # Build data
        if packets_recv == packet_num:
            adc_data.append(buff[buff_pos:buff_pos + packet_length])
            buff_pos += packet_length

        else:  # TODO: HANDLE PACKET REORDERING
            raise ValueError(f'Got packet number {packet_num} but expected {packets_recv}.'
                             f'Current function version does not support out-of-order packet data.')

    adc_data = np.concatenate(adc_data)

    # Write data to destination
    fp = open(dest_fp, 'wb')

    if adc_data.itemsize == 0:
        buffer_size = 0
    else:
        # Set buffer size to 16 MiB to hide the Python loop overhead.
        buffer_size = max(16 * 1024 ** 2 // adc_data.itemsize, 1)

    if adc_data.flags.f_contiguous and not adc_data.flags.c_contiguous:
        for chunk in np.nditer(
                adc_data, flags=['external_loop', 'buffered', 'zerosize_ok'],
                buffersize=buffer_size, order='F'):
            fp.write(chunk.tobytes('C'))
    else:
        for chunk in np.nditer(
                adc_data, flags=['external_loop', 'buffered', 'zerosize_ok'],
                buffersize=buffer_size, order='C'):
            fp.write(chunk.tobytes('C'))

def parse_TSW1400(path, num_chirps_per_frame, num_frames, num_ants, num_adc_samples, iq=True, num_adc_bits=16):
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
        >>> adc_data = parse_TSW1400("./data/radar_data.bin", 128, 200, 4, 256)
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

def parse_DCA1000(path, num_frames, num_chirps_per_frame, num_physical_receivers, num_adc_samples):
    """Parse the DCA1000 ADC binary file into ndarray.

    Given the file path of the ADC binary file from DCA1000, parse it to the numpy ndarray with the shape of
    (num_frames, num_chirps_per_frame, num_physical_receivers, num_adc_samples).

    Args:
        path (string): Path to the binary file.
        num_frames (int): Number of frames captured in the binary data.
        num_chirps_per_frame (int): Number of chirps per frame.
        num_physical_receivers (int): Number of physical receivers used in the ADC data capture.
        num_adc_samples (int): Number of ADC samples.
    
    Returns:
        adc_data (~numpy.ndarray): Organized ADC data in the shape of (num_frames, num_chirps_per_frame, num_physical_receivers,
             num_adc_samples). Currently the only default choice is complex number. Will add the option for real data only.
    """
    adc_data = np.fromfile(path, dtype=np.int16)   
    adc_data = adc_data.reshape(num_frames, -1)
    adc_data = np.apply_along_axis(DCA1000.organize,
                                   1,
                                   adc_data,
                                   num_chirps=num_chirps_per_frame,
                                   num_rx=num_physical_receivers,
                                   num_samples=num_adc_samples)
    
    # first *2 is for complex number and second *2 is for 2 bytes per int16.
    assert adc_data.size*2*2 == os.path.getsize(path), \
        "ndarray size ({}) does not match with file size {}.".format(adc_data.size*2*2, os.path.getsize(path))

    return adc_data
