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
from matplotlib import pyplot as plt
from numpy import pi
from numpy.fft import fft, fftfreq, fftshift
from scipy import signal
import logging
import sys


class ZoomFFT:
    """This class is an implementation of the Zoom Fast Fourier Transform (ZoomFFT).

    The zoom FFT (Fast Fourier Transform) is a signal processing technique used to 
    analyse a portion of a spectrum at high resolution. The steps to apply the zoom 
    FFT to this region are as follows:

    1. Frequency translate to shift the frequency range of interest down to near 
       0 Hz (DC)
    2. Low pass filter to prevent aliasing when subsequently sampled at a lower 
       sample rate
    3. Re-sample at a lower rate
    4. FFT the re-sampled data (Multiple blocks of data are needed to have an FFT of 
       the same length)

    The resulting spectrum will now have a much smaller resolution bandwidth, compared
    to an FFT of non-translated data.

    """

    def __init__(self, low_freq, high_freq, fs, signal=None):
        """Initialize the ZoomFFT class.

        Args:
            low_freq (int): Lower frequency limit
            high_freq (int): Upper frequency limit
            fs (int): sampling Frequency
            signal (np.ndarray): Signal to perform the ZoomFFT on

        """
        self.low_freq = low_freq
        self.high_freq = high_freq
        self.fs = fs

        if (low_freq < 0) or (high_freq > fs) or ((high_freq - low_freq) > fs):
            raise Exception("invalid inputs. Program Terminated! ")

        if signal:
            self.signal = signal
            self.length = len(signal)
        else:
            # the default now is a sine signal, for demo purpose
            pass

    def set_signal(self, signal):
        """Sets given signal as a member variable of the class.

        e.g. ZoomFFT.create_signal(generate_sinewave(a, b, c) + generate_sinewave(d, e, f))

        Args:
            signal (np.ndarray): Signal to perform the ZoomFFT on

        """
        self.signal = signal

    def sinewave(self, f, length, amplitude=1):
        """Generates a sine wave which could be used as a part of the signal. 

        Args: 
            f (int): Frequency of the sine wave
            length (int): Number of data points in the sine wave
            amplitude (int): Amplitude of the sine wave

        Returns:
            x (np.ndarray): Generated sine wave with the given parameters.
        """
        self.length = length
        x = amplitude * np.sin(2 * pi * f / self.fs * np.arange(length))
        return x

    def compute_fft(self):
        """Computes the Fast Fourier Transform (FFT) of the signal.

        Returns:
            X (np.ndarray): A frequency-shifted, unscaled, FFT of the signal.
        """
        try:
            X = fft(self.signal)
            X = np.abs(fftshift(X))  # unscaled
            return X
        except NameError:
            print("signal not defined. Program terminated!")
        except:
            print("Unexpected error:", sys.exc_info()[0])
            raise

    def plot_fft(self, d=None):
        """Plots the Fast Fourier Transform (FFT) of the signal.

        Args:
            d (int): Sample spacing (inverse of the sampling rate)

        """
        try:
            d = 1 / self.fs if d is None else d
            X = self.compute_fft()
            freq = fftfreq(self.length, d)

            self.original_sample_range = 1 / (self.length * d)

            fig1, ax1 = plt.subplots()
            ax1.stem(fftshift(freq), X / self.length)
            ax1.set_xlabel('Frequency (Hz)', fontsize=12)
            ax1.set_ylabel('Magnitude', fontsize=12)
            ax1.set_title('FFT Two-sided spectrum', fontsize=12)
            ax1.grid()

            plt.show()
        except:
            print("Unexpected error:", sys.exc_info()[0])
            raise

    def compute_zoomfft(self, resample_number=None):
        """Computes the Zoom Fast Fourier Transform (ZoomFFT) of the signal.

        Args:
            resample_number (int): The number of samples in the resampled signal.

        Returns:
            Xd (np.ndarray): A frequency-shifted, unscaled, ZoomFFT of the signal.
            bw_factor (int): Bandwidth factor
            fftlen (int): Length of the ZoomFFT output
            Ld (int): for internal use
            F (int): for internal use
        """
        try:
            bw_of_interest = self.high_freq - self.low_freq

            if self.length % bw_of_interest != 0:
                logging.warning("length of signal should be divisible by bw_of_interest. Zoom FFT Spectrum may distort!")
                input("Press Enter to continue...")

            fc = (self.low_freq + self.high_freq) / 2
            bw_factor = np.floor(self.fs / bw_of_interest).astype(np.uint8)

            # mix the signal down to DC, and filter it through the FIR decimator
            ind_vect = np.arange(self.length)
            y = self.signal * np.exp(-1j * 2 * pi * ind_vect * fc / self.fs)

            resample_number = bw_of_interest / self.original_sample_range if resample_number is None else resample_number

            resample_range = bw_of_interest / resample_number

            if resample_range != self.original_sample_range:
                logging.warning("resample resolution != original sample resolution. Zoom FFT Spectrum may distort!")
                input("Press Enter to continue...")

            xd = signal.resample(y, np.int(resample_number))

            fftlen = len(xd)
            Xd = fft(xd)
            Xd = np.abs(fftshift(Xd))  # unscaled

            Ld = self.length / bw_factor
            fsd = self.fs / bw_factor
            F = fc + fsd / fftlen * np.arange(fftlen) - fsd / 2
            return Xd, bw_factor, fftlen, Ld, F
        except NameError:
            print("signal not defined. Program terminated!")
        except:
            print("Unexpected error:", sys.exc_info()[0])
            raise

    def plot_zoomfft(self, resample_number=None):
        """Plots the Zoom Fast Fourier Transform (ZoomFFT) of the signal.

        Args:
            resample_number (int): The number of samples in the resampled signal.
            
        """
        try:
            bw_of_interest = self.high_freq - self.low_freq
            resample_number = bw_of_interest / self.original_sample_range if resample_number is None else resample_number
            Xd, bw_factor, fftlen, Ld, F = self.compute_zoomfft(resample_number)

            fig1, ax1 = plt.subplots()

            ax1.stem(F, Xd / Ld, linefmt='C1-.', markerfmt='C1s')
            ax1.grid()
            ax1.set_xlabel('Frequency (Hz)', fontsize=12)
            ax1.set_ylabel('Magnitude', fontsize=12)
            ax1.set_title('Zoom FFT Spectrum. Mixer Approach.', fontsize=12)
            fig1.subplots_adjust(hspace=0.35)
            plt.show()
        except:
            print("Unexpected error:", sys.exc_info()[0])
            raise
