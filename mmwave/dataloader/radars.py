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
import serial
import struct
import time

MAGIC_WORD_ARRAY = np.array([2, 1, 4, 3, 6, 5, 8, 7])
MAGIC_WORD = b'\x02\x01\x04\x03\x06\x05\x08\x07'
MSG_DETECTED_POINTS = 1
MSG_RANGE_PROFILE = 2
MSG_NOISE_PROFILE = 3
MSG_AZIMUT_STATIC_HEAT_MAP = 4
MSG_POINT_CLOUD_2D = 6


class TI:
    """Software interface to a TI mmWave EVM for reading TLV format. Based on TI's SDKs

    Attributes:
        sdk_version: Version of the TI SDK the radar is using
        cli_port: Serial communication port of the configuration/user port
        data_port: Serial communication port of the data port
        num_rx_ant: Number of RX (receive) antennas being utilized by the radar
        num_tx_ant: Number of TX (transmit) antennas being utilized by the radar
        num_virtual_ant: Number of VX (virtual) antennas being utilized by the radar
        verbose: Optional output messages while parsing data
        connected: Optional attempt to connect to the radar during initialization
        mode: Demo mode to read different TLV formats

    """

    def __init__(self, sdk_version=2.0, cli_loc='COM6', cli_baud=115200,
                 data_loc='COM5', data_baud=921600, num_rx=4, num_tx=2,
                 verbose=False, connect=True, mode=0):
        super(TI, self).__init__()
        self.connected = False
        self.verbose = verbose
        self.mode = mode
        if connect:
            self.cli_port = serial.Serial(cli_loc, cli_baud)
            self.data_port = serial.Serial(data_loc, data_baud)
            self.connected = True
        self.sdk_version = sdk_version
        self.num_rx_ant = num_rx
        self.num_tx_ant = num_tx
        self.num_virtual_ant = num_rx * num_tx
        if mode == 0:
            self._initialize()

    def _configure_radar(self, config):
        for i in config:
            self.cli_port.write((i + '\n').encode())
            print(i)
            time.sleep(0.01)

    def _initialize(self, config_file='./1642config.cfg'):
        config = [line.rstrip('\r\n') for line in open(config_file)]
        if self.connected:
            self._configure_radar(config)

        self.config_params = {}  # Initialize an empty dictionary to store the configuration parameters

        for i in config:

            # Split the line
            split_words = i.split(" ")

            # Hard code the number of antennas, change if other configuration is used
            num_rx_ant = 4
            num_tx_ant = 2

            # Get the information about the profile configuration
            if "profileCfg" in split_words[0]:
                start_freq = int(split_words[2])
                idle_time = int(split_words[3])
                ramp_end_time = float(split_words[5])
                freq_slope_const = int(split_words[8])
                num_adc_samples = int(split_words[10])
                num_adc_samples_round_to2 = 1

                while num_adc_samples > num_adc_samples_round_to2:
                    num_adc_samples_round_to2 = num_adc_samples_round_to2 * 2

                dig_out_sample_rate = int(split_words[11])

            # Get the information about the frame configuration    
            elif "frameCfg" in split_words[0]:

                chirp_start_idx = int(split_words[1])
                chirp_end_idx = int(split_words[2])
                num_loops = int(split_words[3])
                num_frames = int(split_words[4])
                frame_periodicity = float(split_words[5])

        # Combine the read data to obtain the configuration parameters
        num_chirps_per_frame = (chirp_end_idx - chirp_start_idx + 1) * num_loops
        self.config_params["numDopplerBins"] = num_chirps_per_frame / num_tx_ant
        self.config_params["numRangeBins"] = num_adc_samples_round_to2
        self.config_params["rangeResolutionMeters"] = (3e8 * dig_out_sample_rate * 1e3) / (
                2 * freq_slope_const * 1e12 * num_adc_samples)
        self.config_params["rangeIdxToMeters"] = (3e8 * dig_out_sample_rate * 1e3) / (
                2 * freq_slope_const * 1e12 * self.config_params["numRangeBins"])
        self.config_params["dopplerResolutionMps"] = 3e8 / (
                2 * start_freq * 1e9 * (idle_time + ramp_end_time) * 1e-6 * self.config_params[
                    "numDopplerBins"] * num_tx_ant)
        self.config_params["maxRange"] = (300 * 0.9 * dig_out_sample_rate) / (2 * freq_slope_const * 1e3)
        self.config_params["maxVelocity"] = 3e8 / (
                    4 * start_freq * 1e9 * (idle_time + ramp_end_time) * 1e-6 * num_tx_ant)

    def close(self):
        """End connection between radar and machine

        Returns:
            None

        """
        self.cli_port.write('sensorStop\n'.encode())
        self.cli_port.close()
        self.data_port.close()

    def _read_buffer(self):
        """

        Returns:

        """
        byte_buffer = self.data_port.read(self.data_port.in_waiting)
        return byte_buffer

    def _parse_header_data(self, byte_buffer, idx):
        """Parses the byte buffer for the header of the data

        Args:
            byte_buffer: Buffer with TLV data
            idx: Current reading index of the byte buffer

        Returns:
            Tuple [Tuple (int), int]

        """
        magic, idx = self._unpack(byte_buffer, idx, order='>', items=1, form='Q')
        if self.mode == 0:
            (version, length, platform, frame_num, cpu_cycles, num_obj, num_tlvs), idx = self._unpack(byte_buffer, idx,
                                                                                                      items=7, form='I')
            if self.sdk_version > 1.2:
                subframe_num, idx = self._unpack(byte_buffer, idx, items=1, form='I')
            return (version, length, platform, frame_num, cpu_cycles, num_obj, num_tlvs, subframe_num), idx
        else:
            head_1, idx = self._unpack(byte_buffer, idx, items=10, form='I')
            head_2, idx = self._unpack(byte_buffer, idx, items=2, form='H')
            return (*head_1, *head_2), idx

    def _parse_header_tlv(self, byte_buffer, idx):
        """ Parses the byte buffer for the header of a tlv

        """
        (tlv_type, tlv_length), idx = self._unpack(byte_buffer, idx, items=2, form='I')
        return (tlv_type, tlv_length), idx

    def _parse_msg_detected_points(self, byte_buffer, idx):
        """ Parses the information of the detected points message

        """
        (num_detected_points, xyz_qformat), idx = self._unpack(byte_buffer, idx, items=2, form='H')
        range_idx = np.zeros(num_detected_points, dtype=np.int16)
        doppler_idx = np.zeros(num_detected_points, dtype=np.int16)
        peak_val = np.zeros(num_detected_points, dtype=np.int16)
        x = np.zeros(num_detected_points)
        y = np.zeros(num_detected_points)
        z = np.zeros(num_detected_points)

        for i in range(num_detected_points):
            (range_idx[i], doppler_idx[i], peak_val[i]), idx = self._unpack(byte_buffer, idx, items=3, form='H')
            (x[i], y[i], z[i]), idx = self._unpack(byte_buffer, idx, items=3, form='h')

        doppler_idx[doppler_idx > (self.config_params['numDopplerBins'] / 2 - 1)] = doppler_idx[doppler_idx > (
                self.config_params['numDopplerBins'] / 2 - 1)] - 65535
        x *= 1.0 / (1 << xyz_qformat)
        y *= 1.0 / (1 << xyz_qformat)
        z *= 1.0 / (1 << xyz_qformat)

        return (range_idx, doppler_idx, peak_val, x, y, z), idx

    def _parse_msg_azimut_static_heat_map(self, byte_buffer, idx):
        """ Parses the information of the azimuth heat map

        """
        (imag, real), idx = self._unpack(byte_buffer, idx, items=2, form='H')
        return (imag, real), idx

    def _parse_msg_point_cloud_2d(self, byte_buffer, idx):
        """ Parses the information of the 2D point cloud

        """
        (distance, azimuth, doppler, snr), idx = self._unpack(byte_buffer, idx, items=4, form='f')
        return (distance, azimuth, doppler, snr), idx

    def sample(self):
        """ Samples byte data from the radar and converts it to decimal

        """
        byte_buffer = self._read_buffer()

        if len(byte_buffer) < 36:
            return None

        return self._process(byte_buffer)[0]

    def _process(self, byte_buffer):
        """

        """
        all_data = []
        chirps = 0
        while len(byte_buffer) > 32:
            try:
                idx = byte_buffer.index(MAGIC_WORD)
            except:
                return [None] if len(all_data) is 0 else all_data

            header_data, idx = self._parse_header_data(byte_buffer, idx)
            if self.mode == 0:
                data = {'version': header_data[0],
                        'packetLength': header_data[1],
                        'platform': header_data[2],
                        'frameNumber': header_data[3],
                        'timeCpuCycles': header_data[4],
                        'numDetectedObj': header_data[5],
                        'numTLVs': header_data[6],
                        'subFrameNumber': None,
                        'TLVs': []}

                if self.sdk_version > 1.2:
                    data['subFrameNumber'] = header_data[6]

            elif self.mode == 1:
                data = {'version': header_data[0],
                        'platform': header_data[1],
                        'timestamp': header_data[2],
                        'packetLength': header_data[3],
                        'frameNumber': header_data[4],
                        'subframeNumber': header_data[5],
                        'chirpMargin': header_data[6],
                        'frameMargin': header_data[7],
                        'uartSentTime': header_data[8],
                        'trackProcessTime': header_data[9],
                        'numTLVs': header_data[10],
                        'checksum': header_data[11],
                        'TLVs': []}

            for _ in range(data['numTLVs']):
                (tlv_type, tlv_length), idx = self._parse_header_tlv(byte_buffer, idx)
                if len(byte_buffer) < idx + tlv_length:
                    break
                # print('type:', tlv_type)
                if tlv_type == MSG_DETECTED_POINTS:
                    (range_idx, doppler_idx, peak_val, x, y, z), idx = self._parse_msg_detected_points(byte_buffer, idx)
                    data['TLVs'].append(MSG_DETECTED_POINTS)
                elif tlv_type == MSG_AZIMUT_STATIC_HEAT_MAP:
                    azimuth_map = np.zeros((self.num_virtual_ant, self.config_params['numRangeBins'], 2),
                                           dtype=np.int16)
                    for bin_idx in range(self.config_params['numRangeBins']):
                        for ant in range(self.num_virtual_ant):
                            azimuth_map[ant][bin_idx][:], idx = self._parse_msg_azimut_static_heat_map(byte_buffer, idx)
                    data['TLVs'].append(MSG_AZIMUT_STATIC_HEAT_MAP)
                elif tlv_type == MSG_POINT_CLOUD_2D:
                    num_points = tlv_length // 16
                    data['pointCloud2D'] = {}
                    pc = data['pointCloud2D']

                    pc['range'] = np.zeros(num_points, dtype=np.float)
                    pc['azimuth'] = np.zeros(num_points, dtype=np.float)
                    pc['doppler'] = np.zeros(num_points, dtype=np.float)
                    pc['snr'] = np.zeros(num_points, dtype=np.float)
                    for i in range(num_points):
                        (distance, azimuth, doppler, snr), idx = self._parse_msg_point_cloud_2d(byte_buffer, idx)
                        pc['range'][i] = distance
                        pc['azimuth'][i] = azimuth
                        pc['doppler'][i] = doppler
                        pc['snr'][i] = snr
                    data['TLVs'].append(MSG_POINT_CLOUD_2D)
                else:
                    idx += tlv_length

            if MSG_DETECTED_POINTS in data['TLVs']:
                data['rangeIdx'] = range_idx
                data['range'] = range_idx * self.config_params['rangeIdxToMeters']

                data['dopplerIdx'] = doppler_idx
                data['doppler'] = doppler_idx * self.config_params['dopplerResolutionMps']

                data['peakVal'] = peak_val

                data['x'] = x
                data['y'] = y
                data['z'] = z

                # return data 

            elif MSG_AZIMUT_STATIC_HEAT_MAP in data['TLVs']:
                data['azimuthMap'] = np.array(azimuth_map, dtype=np.int32)

                # return data 

            all_data.append(data)
            byte_buffer = byte_buffer[idx:]
            chirps += 1
            if self.verbose and chirps % 100 == 0:
                print('Chirps read:', chirps)

        if self.verbose:
            print('Retrieved data')
        return [None] if len(all_data) is 0 else all_data

    @staticmethod
    def _unpack(byte_buffer, idx, order='', items=1, form='I'):
        """Helper function for parsing binary byte data

        Args:
            byte_buffer: Buffer with data
            idx: Current index in the buffer
            order: Little endian or big endian
            items: Number of items to be extracted
            form: Data type to be extracted

        Returns:
            Tuple [Tuple (object), int]

        """
        size = {'H': 2, 'h': 2, 'I': 4, 'Q': 8, 'f': 4}
        try:
            data = struct.unpack(order + str(items) + form, byte_buffer[idx:idx + (items * size[form])])
            if len(data) == 1:
                data = data[0]
            return data, idx + (items * size[form])
        except:
            return None
