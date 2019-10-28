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

import codecs
import socket
import struct
from enum import Enum

import numpy as np


class CMD(Enum):
    RESET_FPGA_CMD_CODE = '0100'
    RESET_AR_DEV_CMD_CODE = '0200'
    CONFIG_FPGA_GEN_CMD_CODE = '0300'
    CONFIG_EEPROM_CMD_CODE = '0400'
    RECORD_START_CMD_CODE = '0500'
    RECORD_STOP_CMD_CODE = '0600'
    PLAYBACK_START_CMD_CODE = '0700'
    PLAYBACK_STOP_CMD_CODE = '0800'
    SYSTEM_CONNECT_CMD_CODE = '0900'
    SYSTEM_ERROR_CMD_CODE = '0a00'
    CONFIG_PACKET_DATA_CMD_CODE = '0b00'
    CONFIG_DATA_MODE_AR_DEV_CMD_CODE = '0c00'
    INIT_FPGA_PLAYBACK_CMD_CODE = '0d00'
    READ_FPGA_VERSION_CMD_CODE = '0e00'

    def __str__(self):
        return str(self.value)


# MESSAGE = codecs.decode(b'5aa509000000aaee', 'hex')
CONFIG_HEADER = '5aa5'
CONFIG_STATUS = '0000'
CONFIG_FOOTER = 'aaee'
# STATIC
MAX_PACKET_SIZE = 4096


class DCA1000:
    """Software interface to the DCA1000 EVM board via ethernet.

    Attributes:
        static_ip (str): IP to receive data from the FPGA
        adc_ip (str): IP to send configuration commands to the FPGA
        data_port (int): Port that the FPGA is using to send data
        config_port (int): Port that the FPGA is using to read configuration commands from


    General steps are as follows:
        1. Power cycle DCA1000 and XWR1xxx sensor
        2. Open mmWaveStudio and setup normally until tab SensorConfig or use lua script
        3. Make sure to connect mmWaveStudio to the board via ethernet
        4. Start streaming data
        5. Read in frames using class

    Examples:
        >>> dca = DCA1000()
        >>> dca.sensor_config(chirps=128, chirp_loops=3, num_rx=4, num_samples=128)
        >>> adc_data = dca.read(timeout=.001)
        >>> frame = dca.organize(adc_data, 128, 4, 256)

    """

    def __init__(self, static_ip='192.168.33.30', adc_ip='192.168.33.180', data_port=4098, config_port=4096):
        # Save network data
        # self.static_ip = static_ip
        # self.adc_ip = adc_ip
        # self.data_port = data_port
        # self.config_port = config_port

        # Create configuration and data destinations
        self.cfg_dest = (adc_ip, config_port)
        self.cfg_recv = (static_ip, config_port)
        self.data_recv = (static_ip, data_port)

        # Create sockets
        self.config_socket = socket.socket(socket.AF_INET,
                                           socket.SOCK_DGRAM,
                                           socket.IPPROTO_UDP)
        self.data_socket = socket.socket(socket.AF_INET,
                                         socket.SOCK_DGRAM,
                                         socket.IPPROTO_UDP)

        # Bind data socket to fpga
        self.data_socket.bind(self.data_recv)

        # Bind config socket to fpga
        self.config_socket.bind(self.cfg_recv)

        self.lost_packets = None

        # Sensor configuration
        self._bytes_in_frame = None
        self._bytes_in_frame_clipped = None
        self._packets_in_frame = None
        self._packets_in_frame_clipped = None
        self._int16_in_packet = None
        self._int16_in_frame = None
        self.next_frame = None

        # Will be removed in a later release
        self.sensor_config(128, 3, 4, 128)

    def sensor_config(self, chirps, chirp_loops, num_rx, num_samples, iq=2, num_bytes=2):
        """Adjusts the size of the frame returned from realtime reading.

        Args:
            chirps (int): Number of configured chirps in the frame.
            chirp_loops (int): Number of chirp loops per frame.
            num_rx (int): Number of physical receive antennas.
            num_samples (int): Number of samples per chirp.
            iq (int): Number of parts per samples (complex + real).
            num_bytes (int): Number of bytes per part (int16).

        """
        max_bytes_in_packet = 1456  # TODO: WILL CHANGE BASED ON DCA SETTING

        self._bytes_in_frame = chirps * chirp_loops * num_rx * num_samples * iq * num_bytes
        self._bytes_in_frame_clipped = (self._bytes_in_frame // max_bytes_in_packet) * max_bytes_in_packet
        self._packets_in_frame = self._bytes_in_frame / max_bytes_in_packet
        self._packets_in_frame_clipped = self._bytes_in_frame // max_bytes_in_packet
        self._int16_in_packet = max_bytes_in_packet // 2
        self._int16_in_frame = self._bytes_in_frame // 2

        self.next_frame = None

    def configure(self):
        """Initializes and connects to the FPGA

        Returns:
            None

        """
        # SYSTEM_CONNECT_CMD_CODE
        # 5a a5 09 00 00 00 aa ee
        print(self._send_command(CMD.SYSTEM_CONNECT_CMD_CODE))

        # READ_FPGA_VERSION_CMD_CODE
        # 5a a5 0e 00 00 00 aa ee
        print(self._send_command(CMD.READ_FPGA_VERSION_CMD_CODE))

        # CONFIG_FPGA_GEN_CMD_CODE
        # 5a a5 03 00 06 00 01 02 01 02 03 1e aa ee
        print(self._send_command(CMD.CONFIG_FPGA_GEN_CMD_CODE, '0600', 'c005350c0000'))

        # CONFIG_PACKET_DATA_CMD_CODE 
        # 5a a5 0b 00 06 00 c0 05 35 0c 00 00 aa ee
        print(self._send_command(CMD.CONFIG_PACKET_DATA_CMD_CODE, '0600', 'c005350c0000'))

    def close(self):
        """Closes the sockets that are used for receiving and sending data

        Returns:
            None

        """
        self.data_socket.close()
        self.config_socket.close()

    def read(self, timeout=.1):
        """Read in a single packet via UDP

        Args:
            timeout (float): Time to wait for packet before moving on.

        Returns:
            Full frame as array if successful, else None

        """
        # Configure
        self.data_socket.settimeout(timeout)

        # Check if this is the first call
        if self.next_frame is not None:
            ret_frame = self.next_frame
        else:
            ret_frame = np.zeros(self._int16_in_frame, dtype=np.int16)
        self.next_frame = np.zeros(self._int16_in_frame, dtype=np.int16)

        while True:
            packet_num, byte_count, packet_data = self._read_data_packet()
            buff_pointer = ((byte_count // 2) % self._int16_in_frame)

            # Normal packet
            if ((packet_num - 1) % self._packets_in_frame) <= (packet_num % self._packets_in_frame):
                ret_frame[buff_pointer:buff_pointer + packet_data.shape[0]] = packet_data

            # Overflow packet
            else:
                overflow = self._int16_in_frame - buff_pointer
                if buff_pointer > 0:
                    ret_frame[buff_pointer:buff_pointer + overflow] = packet_data[:overflow]
                    self.next_frame[:packet_data.shape[0]-overflow] = packet_data[overflow:]
                else:
                    self.next_frame[:packet_data.shape[0]] = packet_data

                # Try and clear setup next frame before returning
                # while True:
                #     try:
                #         packet_num, byte_count, packet_data = self._read_data_packet()
                #         buff_pointer = ((byte_count // 2) % self._int16_in_frame)
                #         self.next_frame[buff_pointer:buff_pointer + packet_data.shape[0]] = packet_data[:]
                #     except socket.timeout:
                #         break

                return ret_frame

    def _send_command(self, cmd, length='0000', body='', timeout=1):
        """Helper function to send a single commmand to the FPGA

        Args:
            cmd (CMD): Command code to send to the FPGA
            length (str): Length of the body of the command (if any)
            body (str): Body information of the command
            timeout (int): Time in seconds to wait for socket data until timeout

        Returns:
            str: Response message

        """
        # Create timeout exception
        self.config_socket.settimeout(timeout)

        # Create and send message
        resp = ''
        msg = codecs.decode(''.join((CONFIG_HEADER, str(cmd), length, body, CONFIG_FOOTER)), 'hex')
        try:
            self.config_socket.sendto(msg, self.cfg_dest)
            resp, addr = self.config_socket.recvfrom(MAX_PACKET_SIZE)
        except socket.timeout as e:
            print(e)
        return resp

    def _read_data_packet(self):
        """Helper function to read in a single ADC packet via UDP

        Returns:
            int: Current packet number, byte count of data that has already been read, raw ADC data in current packet

        """
        data, addr = self.data_socket.recvfrom(MAX_PACKET_SIZE)
        packet_num = struct.unpack('<1l', data[:4])[0]
        byte_count = struct.unpack('<1Q', data[4:10] + b'\x00\x00')[0]
        packet_data = np.frombuffer(data[10:], dtype=np.int16)
        return packet_num, byte_count, packet_data

    def _listen_for_error(self):
        """Helper function to try and read in for an error message from the FPGA

        Returns:
            None

        """
        self.config_socket.settimeout(None)
        msg = self.config_socket.recvfrom(MAX_PACKET_SIZE)
        if msg == b'5aa50a000300aaee':
            print('stopped:', msg)

    def _stop_stream(self):
        """Helper function to send the stop command to the FPGA

        Returns:
            str: Response Message

        """
        return self._send_command(CMD.RECORD_STOP_CMD_CODE)

    @staticmethod
    def organize(raw_frame, num_chirps, num_rx, num_samples, num_frames=1, model='1642'):
        """Reorganizes raw ADC data into a full frame

        Args:
            raw_frame (ndarray): Data to format.
            num_chirps (int): Number of chirps included in the frame.
            num_rx (int): Number of receivers used in the frame.
            num_samples (int): Number of ADC samples included in each chirp.
            num_frames (int): Number of frames encoded within the data.
            model (str): Model of the radar chip being used.

        Returns:
            ndarray: Reformatted frame of raw data of shape (num_chirps, num_rx, num_samples)

        """
        ret = np.zeros(len(raw_frame) // 2, dtype=np.complex64)

        if model in ['1642', '1843', '6843']:
            # Separate IQ data
            ret[0::2] = raw_frame[0::4] + 1j * raw_frame[2::4]
            ret[1::2] = raw_frame[1::4] + 1j * raw_frame[3::4]
            ret = ret.reshape((num_chirps, num_rx, num_samples)) if num_frames == 1 else ret.reshape(
                (num_frames, num_chirps, num_rx, num_samples))

        elif model in ['1243', '1443']:
            for rx in range(num_rx):
                ret[rx::num_rx] = raw_frame[rx::num_rx * 2] + 1j * raw_frame[rx + num_rx::num_rx * 2]
            ret = ret.reshape((num_chirps, num_samples, num_rx)).swapaxes(1, 2) if num_frames == 1 else ret.reshape(
                (num_frames, num_chirps, num_samples, num_rx)).swapaxes(2, 3)

        else:
            raise ValueError(f'Model {model} is not a supported model')

        return ret
