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

MAX_PACKET_SIZE = 4096
BYTES_IN_PACKET = 1456


def parse_raw_adc(fp):
    buff = np.fromfile(fp, dtype=np.uint8)
    all_data = []
    packets = []
    for num_bytes in range(0, len(buff), BYTES_IN_PACKET):
        data = buff[num_bytes:num_bytes + BYTES_IN_PACKET]
        packet_num = struct.unpack('<1l', data[:4])[0]
        packets.append(packet_num)
        # byte_count = struct.unpack('>Q', b'\x00\x00' + data[4:10][::-1])[0]
        packet_data = np.frombuffer(data[10:], dtype=np.uint16)
        all_data.append(packet_data)

    return all_data, packets
