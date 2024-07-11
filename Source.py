# 光源
import numpy as np


class Source:
    def __init__(self, d2_x, d2_y, wavelength_vacuum, amplitude):
        self.phase = None
        self.k_prop = None
        self.complex_amplitude = None
        self.wavelength_vacuum = wavelength_vacuum
        self.amplitude = amplitude
        self.d2_x = d2_x
        self.d2_y = d2_y

    def plane_wave(self, alpha, beta):
        self.k_prop = 2 * np.pi / self.wavelength_vacuum
        if alpha == np.pi/2 and beta == np.pi/2:
            self.phase = np.zeros_like(self.d2_x)
        else:
            self.phase = self.k_prop * (self.d2_x * np.cos(alpha) + self.d2_y * np.cos(beta))
        self.complex_amplitude = (self.amplitude * np.exp(1j * self.phase))
        return self.complex_amplitude
