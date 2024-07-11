import sys

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
import scipy.interpolate
from scipy import interpolate

from Grid import Grid
from Lens import Lens
from Source import Source
from PropOperator import PropOperator
from Tools import *

save_path = r'E:/Research/WavePropagation/metalens_simulation/Zoom_6×/'  # 数据存放目录

# 建立仿真网络
G = Grid(4096, 2)
# 仿真参数
wavelength_vacuum = 632.8 * 1e-6  # 真空波长，单位mm
refractive_index = 1.457  # SiO2折射率
# 建立透镜
L1 = Lens(G.d2_x, G.d2_y, G.d2_r)
L1.binary2(0.35, 1, 1, [-1.101022155962093e3, 1.561111683794811e2, -4.370083784673994e2,
                        5.311254043289745, 1.319833645746903e4, -4.438157186306786e4])
plt.figure(1, figsize=(9, 7))
plt.pcolormesh(G.d2_x, G.d2_y, L1.phase, cmap="jet")
plt.title('Lens1, Fixation Group')
plt.xlabel(r'$x$(mm)')
plt.ylabel(r'$y$(mm)')
cb = plt.colorbar()
cb.set_label(r'Phase(rad)')  # 给colorbar添加标题
plt.savefig(save_path + 'Lens1.png')
L2 = Lens(G.d2_x, G.d2_y, G.d2_r)
L2.binary2(0.25, 1, 1, [6.135581697936878e3, -7.322451090814901e3, 1.398975370488559e5,
                        -2.930717746220838e6, 2.458970940437622e7, -8.236160880759975e4])
plt.figure(2, figsize=(9, 7))
plt.pcolormesh(G.d2_x, G.d2_y, L2.phase, cmap="jet")
plt.title('Lens2, Zoom Group')
plt.xlabel(r'$x$(mm)')
plt.ylabel(r'$y$(mm)')
cb = plt.colorbar()
cb.set_label(r'Phase(rad)')  # 给colorbar添加标题
plt.savefig(save_path + 'Lens2.png')
L3 = Lens(G.d2_x, G.d2_y, G.d2_r)
L3.binary2(1, 1, 1, [-3.562933126401510e3, 1.769595618032093e2, -8.644704466929417e1,
                     1.352032462467216e2, -1.107469566432938e2, 3.104094666020063e1])
plt.figure(3, figsize=(9, 7))
plt.pcolormesh(G.d2_x, G.d2_y, L3.phase, cmap="jet")
plt.title('Lens3, Compensation Group')
plt.xlabel(r'$x$(mm)')
plt.ylabel(r'$y$(mm)')
cb = plt.colorbar()
cb.set_label(r'Phase(rad)')  # 给colorbar添加标题
plt.savefig(save_path + 'Lens3.png')

# H = Lens(G.d2_x, G.d2_y, G.d2_r)
# H.hole(0.2)
#
# L = Lens(G.d2_x, G.d2_y, G.d2_r)
# L.ideal_lens(0.5, 2, wavelength_vacuum)

# 光源
s = Source(G.d2_x, G.d2_y, wavelength_vacuum, 1)
s.plane_wave(np.pi / 2, np.pi / 2)

# plt.figure(4)
# plt.pcolormesh(G.d2_x, G.d2_y, s.phase, cmap="rainbow")
# plt.colorbar()
# plt.title('Source')
# plt.show()

""" 焦距0.7mm 时 传播 """

# 像面光场计算
d_12 = 9.738333468442362e-1  # 距离，单位mm
d_23 = 3.026166651061564
d_bfl = 1.599999981606190
efl = 0.7  # 有效焦距，单位mm，用于图片命名
p_b = PropOperator(G.d2_fft_x, G.d2_fft_y, wavelength_vacuum, 0.6, refractive_index)  # 基底厚度(base) 传播
p_12 = PropOperator(G.d2_fft_x, G.d2_fft_y, wavelength_vacuum, d_12)  # d_12 传播
p_23 = PropOperator(G.d2_fft_x, G.d2_fft_y, wavelength_vacuum, d_23)  # d_23 传播
p_bfl = PropOperator(G.d2_fft_x, G.d2_fft_y, wavelength_vacuum, d_bfl)  # 后截距(back focal length) 传播
e_1 = p_b.prop(s.complex_amplitude * L1.complex_amplitude_t)
e_2 = p_12.prop(e_1)
e_3 = p_b.prop(e_2 * L2.complex_amplitude_t)
e_4 = p_23.prop(e_3)
e_5 = p_b.prop(e_4 * L3.complex_amplitude_t)
e_6 = p_bfl.prop(e_5)
plt.figure(4, figsize=(16, 7))
plt.subplot(1, 2, 1)
plt.pcolormesh(G.d2_x, G.d2_y, abs(e_6) ** 2, cmap="rainbow")
cb = plt.colorbar()
cb.set_label('Intensity')  # 给colorbar添加标题
plt.title('Interpolation Image')
plt.xlabel(r'$x$(mm)')
plt.ylabel(r'$y$(mm)')
plt.subplot(1, 2, 2)
mid_index = len(G.axis) // 2
plt.plot(G.axis, abs(e_6[mid_index, :]) ** 2)  # 截面图 坐标单位 μm
plt.title(r'$x$-axis Cross Section')
plt.xlabel(r'$x$(mm)')
plt.ylabel('Intensity')
plt.grid()
plt.tight_layout()
plt.savefig(save_path + 'image_f_{:.1f}.png'.format(efl))


# 圈入能量和半高宽计算
ratio_e = calculate_enclosed_energy_ratio(G.axis[mid_index:], abs(e_6[mid_index, mid_index:]) ** 2)
fwhm = calculate_fwhm(G.axis[mid_index:], abs(e_6[mid_index, mid_index:]) ** 2)
print("full width at half maximum is {:.2f} μm".format(fwhm * 1000))
# 对数据进行一倍插值
r_interp = np.linspace(G.axis[mid_index:].min(), G.axis[mid_index:].max(), G.axis.shape[0])
f = interpolate.interp1d(G.axis[mid_index:], ratio_e, kind='cubic')
ratio_e_interp = f(r_interp)

plt.figure(5, figsize=(9, 7))

plt.plot(r_interp * 1000, ratio_e_interp, linewidth=2, color='royalblue')
plt.axvline(x=fwhm / 2 * 1000, color='r', linestyle='--', label='FWHM = {:.2f}μm'.format(fwhm * 1000))
plt.axvline(x=fwhm * 1000, color='orange', linestyle='--', label=r'2FWHM $\eta_f$ = {:.2f}%'.format(f(fwhm) * 100))
plt.axvline(x=fwhm / 2 * 3 * 1000, color='gold', linestyle='--',
            label=r'3FWHM $\eta_f$ = {:.2f}%'.format(f(fwhm / 2 * 3) * 100))
plt.legend()
plt.xlabel(r'$\rho$(μm)')
plt.ylabel('Enclosed Energy Ratio')
plt.xlim(0, 10)
plt.grid()
plt.savefig(save_path + 'enclosed_energy_ratio_f_{:.1f}.png'.format(efl))
plt.show()

# 将结果放大100倍并插值
center_fraction = 0.01  # 中心部分的比例（0 到 1 之间），表示提取的部分大小
d2_x_new = extract_center(G.d2_x, center_fraction)
d2_y_new = extract_center(G.d2_y, center_fraction)
e_f_new = extract_center(abs(e_6) ** 2, center_fraction)

# plt.figure(5)
# plt.pcolormesh(d2_x_new, d2_y_new, e_f_new, cmap="jet")
# plt.colorbar()

# 插值后的新网格
interp_rows = d2_x_new.shape[0] * 2
interp_cols = d2_y_new.shape[1] * 2
f = interpolate.RegularGridInterpolator((d2_x_new[0, :], d2_y_new[:, 0]), e_f_new)
x_interp = np.linspace(d2_x_new.min(), d2_x_new.max(), interp_rows)
y_interp = np.linspace(d2_y_new.min(), d2_y_new.max(), interp_cols)
d2_x_interp, d2_y_interp = np.meshgrid(x_interp, y_interp)
e_f_interp = f((d2_x_interp, d2_y_interp))

# 在新网格上插值

plt.figure(6, figsize=(16, 7))
plt.subplot(1, 2, 1)
plt.pcolormesh(d2_x_interp * 1000, d2_y_interp * 1000, e_f_interp, cmap="jet")  # 插值图 坐标单位 μm
cb = plt.colorbar()
cb.set_label('Intensity')  # 给colorbar添加标题
plt.title('Interpolation Image')
plt.xlabel(r'$x$(μm)')
plt.ylabel(r'$y$(μm)')
plt.subplot(1, 2, 2)
mid_index = len(y_interp) // 2
plt.plot(x_interp * 1000, e_f_interp[mid_index, :])  # 截面图 坐标单位 μm
plt.title(r'$x$-axis Cross Section')
plt.xlabel(r'$x$(μm)')
plt.ylabel('Intensity')
plt.grid()
plt.tight_layout()
plt.savefig(save_path + 'image_interp_image_{:.1f}.png'.format(efl))
plt.show()

# 后截距y-z截面
s = 200  # 取样点数
dist_0 = 1  # 焦点前后的距离,单位 mm
dist_array = np.linspace(d_bfl - dist_0, d_bfl + dist_0, s)
e_yz = np.zeros((len(G.axis), len(dist_array)), dtype=complex)
mid_index = len(G.axis) // 2
for i in range(len(dist_array)):
    p_bfl = PropOperator(G.d2_fft_x, G.d2_fft_y, wavelength_vacuum, dist_array[i])
    e_yz_d = p_bfl.prop(e_5)
    e_yz[:, i] = e_yz_d[mid_index, :]

    progress = (i + 1) / len(dist_array) * 100

    # 输出进度
    sys.stdout.write(f"\rProgress: {progress:.2f}%")
    sys.stdout.flush()  # 输出到屏幕
z, x = np.meshgrid(dist_array, G.axis)
plt.figure(7, figsize=(9, 7))
plt.pcolormesh(z, x, np.log(abs(e_yz) ** 2), cmap="jet", vmin=-5)
plt.title(r'$x-z$ Cross Section')
plt.xlabel(r'$z$(mm)')
plt.ylabel(r'$x$(mm)')
cb = plt.colorbar()
cb.set_label('ln(Intensity)')  # 给colorbar添加标题
plt.savefig(save_path + 'x-z_cross_section_{:.1f}.png'.format(efl))
plt.show()

# 理想透镜测试
# p_f1 = PropOperator(G.d2_fft_x, G.d2_fft_y, wavelength_vacuum, 4,paraxial=False)
# plt.pcolormesh(G.d2_x, G.d2_y, L.phase, cmap="rainbow")
# e_f = p_f1.prop(s.complex_amplitude * L.complex_amplitude_t)
#
# plt.figure(4)
# plt.pcolormesh(G.d2_x, G.d2_y, abs(e_f) ** 2, cmap="rainbow")
# plt.show()
