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
import struct


def parse_raw_adc(source_fp, dest_fp):
    """Reads a binary data file containing raw adc data from a DCA1000, cleans it and saves it for manual processing.

    Note:
        "Raw adc data" in this context refers to the fact that the DCA1000 initially sends packets of data containing
        meta data and is merged with actual pure adc data. Part of the purpose of this function is to remove this
        meta data.

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

        # Zero fill array
        elif packets_recv < packet_num:
            while packets_recv < packet_num:
                adc_data.append(np.zeros(packet_length))
                packets_recv += 1
            adc_data.append(buff[buff_pos:buff_pos + packet_length])
            buff_pos += packet_length

        # Place packet in correct place
        else:
            adc_data[packet_num-1] = buff[buff_pos:buff_pos + packet_length]
            buff_pos += packet_length

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
