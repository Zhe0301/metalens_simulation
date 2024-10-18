# 光源
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams
import matplotlib

config = {"font.family": 'serif',
          "font.size": 20,
          "mathtext.fontset": 'stix',
          "font.serif": ['Times New Roman']
          }
rcParams.update(config)

matplotlib.use('qt5agg')


class Source:
    def __init__(self, Grid, wavelength_vacuum, amplitude):
        self.Grid = Grid
        self.energy = None
        self.phase = None
        self.k_prop = None
        self.complex_amplitude = None
        self.wavelength_vacuum = wavelength_vacuum
        self.amplitude = B = np.full(Grid.d2_x.shape, amplitude)


    def plane_wave(self, alpha, beta):
        # alpha,beta是平面波的方向余弦，弧度制
        self.k_prop = 2 * np.pi / self.wavelength_vacuum
        if alpha == np.pi / 2 and beta == np.pi / 2:
            self.phase = np.zeros_like(self.Grid.d2_x)
        else:
            self.phase = self.k_prop * (self.Grid.d2_x * np.cos(alpha) + self.Grid.d2_y * np.cos(beta))
        self.complex_amplitude = (self.amplitude * np.exp(1j * self.phase))
        return self.complex_amplitude

    def plot_phase(self, save_path=r'/'):
        phase_2pi = np.mod(self.phase, 2 * np.pi)
        plt.figure(figsize=(16, 7))
        plt.subplot(1, 2, 1)
        plt.pcolormesh(self.Grid.d2_x, self.Grid.d2_y, self.phase, cmap="jet")
        plt.title('Phase Distribution')
        plt.xlabel(r'$x$(mm)')
        plt.ylabel(r'$y$(mm)')
        cb = plt.colorbar()
        cb.set_label(r'Phase(rad)')  # 给colorbar添加标题
        plt.subplot(1, 2, 2)
        plt.pcolormesh(self.Grid.d2_x, self.Grid.d2_y, phase_2pi, cmap="jet")
        plt.title(r'Phase Distribution $0 \sim 2\pi$')
        plt.xlabel(r'$x$(mm)')
        plt.ylabel(r'$y$(mm)')
        cb = plt.colorbar()
        cb.set_label(r'Phase(rad)')  # 给colorbar添加标题
        plt.tight_layout()
        if save_path is not None:
            plt.savefig(save_path + 'Source_Phase.png')
        plt.show()
        plt.close()

    def plot_intensity(self, save_path=r'/'):
        plt.figure(figsize=(9, 7))
        plt.pcolormesh(self.Grid.d2_x, self.Grid.d2_y, np.abs(self.complex_amplitude_t) ** 2, cmap="jet")
        plt.title('Intesity')
        plt.xlabel(r'$x$(mm)')
        plt.ylabel(r'$y$(mm)')
        cb = plt.colorbar()
        cb.set_label(r'Intesity')  # 给colorbar添加标题
        if save_path is not None:
            plt.savefig(save_path + 'Source_Intesity.png')
        plt.show()
        plt.close()
