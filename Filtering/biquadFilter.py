import numpy as np

class BiquadFilter:
    def __init__(self, nchan, type, fc, q, peakGainDB):
        self.nchan = nchan
        self.type = type
        self.fc = fc
        self.q = q
        self.peakGain = peakGainDB
        self.a0 = self.a1 = self.a2 = self.b1 = self.b2 = 0.0
        self.z1 = np.zeros(nchan)
        self.z2 = np.zeros(nchan)
        self.setBiquad(type, fc, q, peakGainDB)

    def setBiquad(self, type, fc, q, peakGainDB):
        self.type = type
        self.fc = fc
        self.q = q
        self.peakGain = peakGainDB
        self.calcBiquad()

    def calcBiquad(self):
        PI = np.pi
        k = np.tan(PI * self.fc)
        norm = 0
        if self.type == 0:  # Lowpass
            norm = 1 / (1 + k / self.q + k**2)
            self.a0 = k**2 * norm
            self.a1 = 2 * self.a0
            self.a2 = self.a0
            self.b1 = 2 * (k**2 - 1) * norm
            self.b2 = (1 - k / self.q + k**2) * norm
        elif self.type == 1:  # Highpass
            norm = 1 / (1 + k / self.q + k**2)
            self.a0 = 1 * norm
            self.a1 = -2 * self.a0
            self.a2 = self.a0
            self.b1 = 2 * (k**2 - 1) * norm
            self.b2 = (1 - k / self.q + k**2) * norm
        elif self.type == 2:  # Bandpass
            norm = 1 / (1 + k / self.q + k**2)
            self.a0 = k / self.q * norm
            self.a1 = 0.0
            self.a2 = -self.a0
            self.b1 = 2 * (k**2 - 1) * norm
            self.b2 = (1 - k / self.q + k**2) * norm
        elif self.type == 3:  # Notch
            norm = 1 / (1 + k / self.q + k**2)
            self.a0 = (1 + k**2) * norm
            self.a1 = 2 * (k**2 - 1) * norm
            self.a2 = self.a0
            self.b1 = self.a1
            self.b2 = (1 - k / self.q + k**2) * norm

    def process(self, input, ichan):
        output = input * self.a0 + self.z1[ichan]
        self.z1[ichan] = input * self.a1 + self.z2[ichan] - self.b1 * output
        self.z2[ichan] = input * self.a2 - self.b2 * output
        return output