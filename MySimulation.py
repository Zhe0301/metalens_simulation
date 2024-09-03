"""
建立超透镜模型通用函数
"""
import sys

from matplotlib import pyplot as plt
from matplotlib import rcParams

import matplotlib

config = {"font.family": 'serif',
          "font.size": 20,
          "mathtext.fontset": 'stix',
          "font.serif": ['Times New Roman']
          }
rcParams.update(config)

matplotlib.use('qt5agg')
from scipy.interpolate import interpolate

from PropOperator import PropOperator
import cupy as cp
import time
from Tools import *


def three_element_zoom_system(s, L1, L2, L3, G, d_lens, d_12, d_23, d_bfl, efl, wavelength_vacuum, refractive_index,
                              save_path, magnification=100, sampling_point=200, interval=1, method='AS',
                              gpu_acceleration=False):
    # magnification 图像放大倍率，为0或小于0时不放大,算y-z平面光场（sampling_point采样点数,interval焦点前后距离，单位mm，二者之一为0或小于0时不计算）
    t0 = time.time()
    if gpu_acceleration:  # 使用GPU加速先进行数据转换
        G.d2_fft_x = cp.asarray(G.d2_fft_x)
        G.d2_fft_y = cp.asarray(G.d2_fft_y)
        G.axis = cp.asarray(G.axis)
        L1.complex_amplitude_t = cp.asarray(L1.complex_amplitude_t)
        L1.mask = cp.asarray(L1.mask)
        L2.complex_amplitude_t = cp.asarray(L2.complex_amplitude_t)
        L3.complex_amplitude_t = cp.asarray(L3.complex_amplitude_t)
        s.complex_amplitude = cp.asarray(s.complex_amplitude)
        xp = cp
    else:
        xp = np

    p_b = PropOperator(G, wavelength_vacuum, d_lens, refractive_index, method=method,
                       gpu_acceleration=gpu_acceleration)  # 基底厚度(base) 传播
    p_12 = PropOperator(G, wavelength_vacuum, d_12, method=method, gpu_acceleration=gpu_acceleration)  # d_12 传播
    p_23 = PropOperator(G, wavelength_vacuum, d_23, method=method, gpu_acceleration=gpu_acceleration)  # d_23 传播
    p_bfl = PropOperator(G, wavelength_vacuum, d_bfl, method=method,
                         gpu_acceleration=gpu_acceleration)  # 后截距(back focal length) 传播
    t1 = time.time()
    elapsed_time = t1 - t0
    print("Parameter initialization is completed. Elapsed time: {:.2f} s".format(elapsed_time))
    print("Calculating light propagation. 1/6 ")
    e_1 = p_b.prop(s.complex_amplitude * L1.complex_amplitude_t)
    print("Calculating light propagation. 2/6 ")
    e_2 = p_12.prop(e_1)
    print("Calculating light propagation. 3/6 ")
    e_3 = p_b.prop(e_2 * L2.complex_amplitude_t)
    print("Calculating light propagation. 4/6 ")
    e_4 = p_23.prop(e_3)
    print("Calculating light propagation. 5/6 ")
    e_5 = p_b.prop(e_4 * L3.complex_amplitude_t)
    print("Calculating light propagation. 6/6 ")
    e_6 = p_bfl.prop(e_5)
    t2 = time.time()
    elapsed_time = t2 - t1
    print("Light propagation calculation is completed. Elapsed time: {:.2f} s".format(elapsed_time))
    """MTF计算"""
    mtf_x, mtf_y = calculate_mtf(e_6, G, gpu_acceleration=gpu_acceleration)
    """圈入能量计算"""
    mid_index_0 = len(G.axis) // 2  # 坐标中心位置
    source_energy = xp.sum(xp.power(xp.abs(s.complex_amplitude * L1.mask), 2))  # 光源入射的能量
    image_energy = xp.sum(xp.power(xp.abs(e_6), 2))  # 像面上的能量
    transmission_efficient = image_energy / source_energy  # 传输效率
    print(r"Transmission efficient: {:.2f}%".format(transmission_efficient * 100))
    ratio_e = (calculate_enclosed_energy_ratio(G.axis[mid_index_0:], xp.abs(e_6[mid_index_0, mid_index_0:]) ** 2)
               * transmission_efficient)
    fwhm = calculate_fwhm(G.axis[mid_index_0:], xp.abs(e_6[mid_index_0, mid_index_0:]) ** 2)
    print("full width at half maximum is {:.2f} μm".format(fwhm * 1000))

    t3 = time.time()

    """后截距的y-z截面计算"""
    if interval > 0 and sampling_point > 0:
        print("Calculating the light field in y-z cross section")
        if gpu_acceleration:
            e_5 = cp.asarray(e_5)
        sp = sampling_point  # 取样点数
        dist_0 = 1  # 焦点前后的距离,单位 mm
        dist_array = np.linspace(d_bfl - dist_0, d_bfl + dist_0, sp)
        e_yz = xp.zeros((len(G.axis), len(dist_array)), dtype=complex)
        for i in range(len(dist_array)):
            p_bfl = PropOperator(G.d2_fft_x, G.d2_fft_y, wavelength_vacuum, dist_array[i])
            e_yz_d = p_bfl.prop(e_5)
            e_yz[:, i] = e_yz_d[mid_index_0, :]
            progress = (i + 1) / len(dist_array) * 100
            # 输出进度
            sys.stdout.write(f"\rProgress: {progress:.2f}%")
            sys.stdout.flush()  # 输出到屏幕
        t4 = time.time()
        elapsed_time = t4 - t3
        print("\n y-z cross section calculation is completed. Elapsed time: {:.2f} s".format(elapsed_time))
        if gpu_acceleration:
            e_yz = cp.asnumpy(e_yz)

    if gpu_acceleration:
        e_5 = cp.asnumpy(e_5)
        e_6 = cp.asnumpy(e_6)
        G.d2_fft_x = cp.asnumpy(G.d2_fft_x)
        G.d2_fft_y = cp.asnumpy(G.d2_fft_y)
        G.axis = cp.asnumpy(G.axis)
        ratio_e = cp.asnumpy(ratio_e)
        fwhm = cp.asnumpy(fwhm)
        mtf_x = cp.asnumpy(mtf_x)
        mtf_y = cp.asnumpy(mtf_y)

    """统一绘图"""
    plt.figure(figsize=(9, 7))
    plt.title('MTF')
    plt.xlabel('frequency(/mm)')
    plt.ylabel(r'MTF')
    plt.plot(G.axis_fft,mtf_x,G.axis_fft,mtf_y)
    plt.show()

    plt.figure(figsize=(16, 14))
    plt.subplot(2, 2, 1)
    plt.pcolormesh(G.d2_x, G.d2_y, np.abs(e_6) ** 2, cmap="rainbow")
    cb = plt.colorbar()
    cb.set_label('Intensity')  # 给colorbar添加标题
    plt.title('Image')
    plt.xlabel(r'$x$(mm)')
    plt.ylabel(r'$y$(mm)')
    plt.subplot(2, 2, 2)
    plt.plot(G.axis, np.abs(e_6[mid_index_0, :]) ** 2)  # 截面图 坐标单位 μm
    plt.title(r'$x$-axis Cross Section')
    plt.xlabel(r'$x$(mm)')
    plt.ylabel('Intensity')
    plt.grid()
    plt.subplot(2, 2, 3)
    plt.pcolormesh(G.d2_x, G.d2_y, np.log(np.abs(e_6) * 2), cmap="rainbow")
    cb = plt.colorbar()
    cb.set_label('ln(Intensity)')  # 给colorbar添加标题
    plt.title('Image')
    plt.xlabel(r'$x$(mm)')
    plt.ylabel(r'$y$(mm)')
    plt.subplot(2, 2, 4)
    plt.plot(G.axis, np.log(np.abs(e_6[mid_index_0, :])) * 2)  # 截面图 坐标单位 μm
    plt.title(r'$x$-axis Cross Section')
    plt.xlabel(r'$x$(mm)')
    plt.ylabel('Intensity')
    plt.grid()
    plt.tight_layout()
    plt.savefig(save_path + 'image_f_{:.1f}.png'.format(efl))

    # 放大像面并插值绘图
    t3 = time.time()
    if magnification > 0:
        # 将结果放大并插值
        center_fraction = 1 / magnification  # 中心部分的比例（0 到 1 之间），表示提取的部分大小
        d2_x_new = extract_center(G.d2_x, center_fraction)
        d2_y_new = extract_center(G.d2_y, center_fraction)
        e_f_new = extract_center(abs(e_6) ** 2, center_fraction)

        # plt.figure(5)
        # plt.pcolormesh(d2_x_new, d2_y_new, e_f_new, cmap="jet")
        # plt.colorbar()

        # 插值后的新网格
        interp_rows = d2_x_new.shape[0] * 2
        interp_cols = d2_y_new.shape[1] * 2
        f = interpolate.RegularGridInterpolator((d2_x_new[0, :], d2_y_new[:, 0]), e_f_new, method='linear')
        x_interp = np.linspace(d2_x_new.min(), d2_x_new.max(), interp_rows)
        y_interp = np.linspace(d2_y_new.min(), d2_y_new.max(), interp_cols)
        d2_x_interp, d2_y_interp = np.meshgrid(x_interp, y_interp)
        e_f_interp = f((d2_x_interp, d2_y_interp))
        # 在新网格上插值
        plt.figure(figsize=(16, 7))
        plt.subplot(1, 2, 1)
        plt.pcolormesh(d2_x_interp * 1000, d2_y_interp * 1000, e_f_interp, cmap="jet")  # 插值图 坐标单位 μm
        cb = plt.colorbar()
        cb.set_label('Intensity')  # 给colorbar添加标题
        plt.title('Interpolation Image')
        plt.xlabel(r'$x$(μm)')
        plt.ylabel(r'$y$(μm)')
        plt.subplot(1, 2, 2)
        mid_index_1 = len(y_interp) // 2
        plt.plot(x_interp * 1000, e_f_interp[mid_index_1, :])  # 截面图 坐标单位 μm
        plt.title(r'$x$-axis Cross Section')
        plt.xlabel(r'$x$(μm)')
        plt.ylabel('Intensity')
        plt.grid()
        plt.tight_layout()
        plt.savefig(save_path + 'image_interp_image_{:.1f}.png'.format(efl))

    # 圈入能量和半高宽插值绘图
    r_interp = np.linspace(G.axis[mid_index_0:].min(), G.axis[mid_index_0:].max(), G.axis.shape[0])  # 对数据进行一倍插值
    f = interpolate.interp1d(G.axis[mid_index_0:], ratio_e, kind='cubic')
    ratio_e_interp = f(r_interp)

    plt.figure(figsize=(9, 7))

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

    t4 = time.time()
    # y-z平面绘图
    if interval > 0 and sampling_point > 0:
        z, x = np.meshgrid(dist_array, G.axis)
        t5 = time.time()
        plt.figure(figsize=(9, 7))
        plt.pcolormesh(z, x, np.log(abs(e_yz) ** 2), cmap="jet", vmin=-5)
        plt.title(r'$x-z$ Cross Section')
        plt.xlabel(r'$z$(mm)')
        plt.ylabel(r'$x$(mm)')
        cb = plt.colorbar()
        cb.set_label('ln(Intensity)')  # 给colorbar添加标题
        plt.savefig(save_path + 'x-z_cross_section_{:.1f}.png'.format(efl))
        plt.show()
        return e_5, e_6, e_yz
    else:
        return e_5, e_6
