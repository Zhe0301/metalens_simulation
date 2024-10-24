"""
建立变焦超透镜模型通用函数
3片式变焦系统
超透镜位于每片的后表面
20240904
周王哲
"""
import sys
import h5py
from matplotlib import pyplot as plt
from matplotlib import rcParams
from loguru import logger
import matplotlib
from tqdm import tqdm

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


def three_element_zoom_system(S, L1, L2, L3, G, d_lens, d_12, d_23, d_bfl, efl, refractive_index,
                              save_path, magnification=100, sampling_point=200, interval=1, method='AS', show=False,
                              gpu_acceleration=False):
    """
    S 光源对象
    L1 L2 L3 三片镜片对象
    G 网络对象
    d_lens d_12 d_23 d_bfl 镜片厚度 1，2片间距 2，3片间距 后截距
    efl 有效焦距，用于文件命名
    refractive_index 镜片折射率
    save_path 文件储存目录
    magnification 图像放大倍率，为0或小于0时不放大,算y-z平面光场（sampling_point采样点数,interval焦点前后距离，单位mm，二者之一为0或小于0时不计算）
    sampling_point x-z截面计算的采样点数，为0时不计算
    interval x-z截面采样时焦点前后的距离，为0时不计算，单位mm
    """
    if gpu_acceleration:
        logger.info("Using GPU to accelerate computing")
    t0 = time.time()

    p_b = PropOperator(G, S.wavelength_vacuum, d_lens, refractive_index = refractive_index, method=method,
                       gpu_acceleration=gpu_acceleration)  # 基底厚度(base) 传播
    p_12 = PropOperator(G, S.wavelength_vacuum, d_12, method=method, gpu_acceleration=gpu_acceleration)  # d_12 传播
    p_23 = PropOperator(G, S.wavelength_vacuum, d_23, method=method, gpu_acceleration=gpu_acceleration)  # d_23 传播
    p_bfl = PropOperator(G, S.wavelength_vacuum, d_bfl, method=method,
                         gpu_acceleration=gpu_acceleration)  # 后截距(back focal length) 传播
    t1 = time.time()
    elapsed_time = t1 - t0
    logger.success("Parameter initialization is completed. Elapsed time: {:.2f} s".format(elapsed_time))
    logger.info("Calculating light propagation. 1/6 ")
    e_1 = p_b.prop(S.complex_amplitude)
    logger.info("Calculating light propagation. 2/6 ")
    e_2 = p_12.prop(e_1 * L1.complex_amplitude_t)
    del e_1
    logger.info("Calculating light propagation. 3/6 ")
    e_3 = p_b.prop(e_2)
    del e_2
    logger.info("Calculating light propagation. 4/6 ")
    e_4 = p_23.prop(e_3 * L2.complex_amplitude_t)
    del e_3
    logger.info("Calculating light propagation. 5/6 ")
    e_5 = p_b.prop(e_4)
    del e_4
    logger.info("Calculating light propagation. 6/6 ")
    e_5 = e_5 * L3.complex_amplitude_t
    e_6 = p_bfl.prop(e_5)
    t2 = time.time()
    elapsed_time = t2 - t1
    logger.success("Light propagation calculation is completed. Elapsed time: {:.2f} s".format(elapsed_time))

    """MTF计算"""
    mtf_x, mtf_y, psf = calculate_mtf(np.power(np.abs(e_6), 2), G.step, gpu_acceleration=gpu_acceleration)
    # MTF绘图
    plt.figure(figsize=(16, 7))
    plt.subplot(1, 2, 1)
    plt.title('MTF')
    plt.xlabel('Frequency(lp/mm)')
    plt.ylabel(r'MTF')
    plt.plot(G.axis_fft[len(mtf_x):], mtf_x,linewidth=2,label=r'$x$')
    plt.plot(G.axis_fft[len(mtf_x):], mtf_y,linewidth=2,label=r'$y$',linestyle="--")
    plt.grid()
    plt.legend()
    plt.subplot(1, 2, 2)
    plt.title('PSF')
    plt.imshow(psf, cmap="rainbow")
    cb = plt.colorbar()
    cb.set_label('Intensity')  # 给colorbar添加标题
    plt.tight_layout()
    plt.savefig(save_path + 'MTF_f_{:.1f}.png'.format(efl))
    if show:
        plt.show()
    plt.close()
    """圈入能量计算"""
    mid_index_0 = len(G.axis) // 2  # 坐标中心位置
    source_energy = np.sum(np.power(np.abs(S.complex_amplitude * L1.mask), 2))  # 光源入射的能量
    image_energy = np.sum(np.power(np.abs(e_6), 2))  # 像面上的能量
    transmission_efficient = image_energy / source_energy  # 传输效率
    logger.info(r"Transmission efficient: {:.2f}%".format(transmission_efficient * 100))
    ratio_e = (calculate_enclosed_energy_ratio(G.axis[mid_index_0:], np.abs(e_6[mid_index_0, mid_index_0:]) ** 2))
    fwhm = calculate_fwhm(G.axis[mid_index_0:], np.abs(e_6[mid_index_0, mid_index_0:]) ** 2)
    logger.info("full width at half maximum is {:.2f} μm".format(fwhm * 1000))

    # 像面绘图
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
        plt.savefig(save_path + 'image_interp_f_{:.1f}.png'.format(efl))

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
    if show:
        plt.show()
    plt.close()

    t3 = time.time()

    """后截距的y-z截面计算"""
    if interval > 0 and sampling_point > 0:
        logger.info("Calculating the light field in y-z cross section")
        dist_array = np.linspace(d_bfl - interval, d_bfl + interval, sampling_point)
        e_yz = np.zeros((len(G.axis), len(dist_array)), dtype=complex)
        for i in tqdm(range(len(dist_array))):
            p_bfl = PropOperator(G, S.wavelength_vacuum, dist_array[i], method=method,
                         gpu_acceleration=gpu_acceleration)
            e_yz_d = p_bfl.prop(e_5)
            e_yz[:, i] = e_yz_d[mid_index_0, :]

        t4 = time.time()
        elapsed_time = t4 - t3
        logger.info("\n y-z cross section calculation is completed. Elapsed time: {:.2f} s".format(elapsed_time))

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
        if show:
            plt.show()
        plt.close()
        with h5py.File(save_path + "e_5_f_{:.1f}.h5".format(efl), 'w') as f:
            f.create_dataset('complex_amplitude', data=e_5, compression='gzip', compression_opts=9)
        with h5py.File(save_path + 'e_6_f_{:.1f}.h5'.format(efl), 'w') as f:
            f.create_dataset('complex_amplitude', data=e_6, compression='gzip', compression_opts=9)
        with h5py.File(save_path + 'e_yz_f_{:.1f}.h5'.format(efl), 'w') as f:
            f.create_dataset('complex_amplitude', data=e_yz, compression='gzip', compression_opts=9)
        return e_5, e_6, e_yz
    else:
        with h5py.File(save_path + "e_5_f_{:.1f}.h5".format(efl), 'w') as f:
            f.create_dataset('complex_amplitude', data=e_5, compression='gzip', compression_opts=9)
        with h5py.File(save_path + 'e_6_f_{:.1f}.h5'.format(efl), 'w') as f:
            f.create_dataset('complex_amplitude', data=e_6, compression='gzip', compression_opts=9)
        return e_5, e_6
