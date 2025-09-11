'''
202507
Add noise to nifti volumes, avoiding coordinates(MNI space)
'''
import nibabel as nib
import numpy as np
from scipy.ndimage import binary_dilation
from scipy.signal import butter, filtfilt
from pathlib import Path
from tqdm import tqdm
import pdb

mni_coords = [  # avoid DMN, SN, CEN nodes
    (-18, 24, 53),
    (22, 26, 51),
    (-18, -1, 65),
    (20, 4, 64),
    (-27, 43, 31),
    (30, 37, 36),
    (-42, 13, 36),
    (42, 11, 39),
    (-28, 56, 12),
    (28, 55, 17),
    (-41, 41, 16),
    (42, 44, 14),
    (-33, 23, 45),
    (42, 27, 39),
    (-32, 4, 55),
    (34, 8, 54),
    (-26, 60, -6),
    (25, 61, -4),
    (-65, -30, -12),
    (65, -29, -13),
    (-53, 2, -30),
    (51, 6, -32),
    (-59, -58, 4),
    (60, -53, 3),
    (-58, -20, -9),
    (58, -16, -10),
    (-27, -7, -34),
    (28, -8, -33),
    (-25, -25, -26),
    (26, -23, -27),
    (-28, -32, -18),
    (30, -30, -18),
    (-19, -12, -30),
    (19, -10, -30),
    (-23, 2, -32),
    (22, 1, -36),
    (-17, -39, -10),
    (19, -36, -11),
    (-16, -60, 63),
    (19, -57, 65),
    (-27, -59, 54),
    (31, -54, 53),
    (-34, -80, 29),
    (45, -71, 20),
    (-38, -61, 46),
    (39, -65, 44),
    (-51, -33, 42),
    (47, -35, 45),
    (-56, -49, 38),
    (57, -44, 38),
    (-47, -65, 26),
    (53, -54, 25),
    (-53, -31, 23),
    (55, -26, 26),
    (-5, -63, 51),
    (6, -65, 51),
    (-8, -47, 57),
    (7, -47, 58),
    (-12, -67, 25),
    (16, -64, 25),
    (-6, -55, 34),
    (6, -54, 35),
    (-36, -20, 10),
    (37, -18, 8),
    (-32, 14, -13),
    (33, 14, -13),
    (-34, 18, 1),
    (36, 18, 1),
    (-38, -4, -9),
    (39, -2, -9),
    (-38, -8, 8),
    (39, -7, 8),
    (-38, 5, 5),
    (38, 5, 5),
    (-4, -39, 31),
    (4, -37, 32),
    (-3, 8, 25),
    (5, 22, 12),
    (-6, 34, 21),
    (5, 28, 27),
    (-8, -47, 10),
    (9, -44, 11),
    (-5, 7, 37),
    (4, 6, 38),
    (-7, -23, 41),
    (6, -20, 40),
    (-4, 39, -2),
    (5, 41, 6),
    (-19, -2, -20),
    (19, -2, -19),
    (-27, -4, -20),
    (28, -3, -20),
    (-22, -14, -19),
    (22, -12, -20),
    (-28, -30, -10),
    (29, -27, -10),
    (-7, -12, 5),
    (7, -11, 6),
    (-18, -13, 3),
    (12, -14, 1),
    (-18, -23, 4),
    (18, -22, 3),
    (-7, -14, 7),
    (3, -13, 5),
    (-16, -24, 6),
    (15, -25, 6),
    (-15, -28, 4),
    (13, -27, 8),
    (-12, -22, 13),
    (10, -14, 14),
    (-11, -14, 2),
    (13, -16, 7)
    ]

def add_noise_avoid_coordinates(nifti_file, mni_coordinates, 
                               output_file=None, 
                               avoid_radius=5, 
                               noise_type='gaussian',
                               noise_params=None,
                               volumes=None,
                               save_mask=False,
                               use_brain_mask=False,
                               use_band_filter=True):
    """
    给NIfTI数据添加噪声，但绕过指定的MNI坐标区域
    
    参数:
    - nifti_file: 输入NIfTI文件路径
    - mni_coordinates: MNI坐标列表 [(x1,y1,z1), (x2,y2,z2), ...]
    - output_file: 输出文件路径
    - avoid_radius: 绕过半径(mm)
    - noise_type: 噪声类型 ('gaussian', 'uniform', 'salt_pepper', 'multiplicative')
    - noise_params: 噪声参数字典
    - volumes: 指定处理的volume索引，None表示处理所有
    - save_mask: 是否保存避开区域的mask
    
    返回:
    - noisy_data: 添加噪声后的数据
    - protection_mask: 保护区域的mask
    """
    
    # 加载数据
    img = nib.load(nifti_file)
    data = img.get_fdata()
    affine = img.affine
    
    print(f"原始数据形状: {data.shape}")
    print(f"要保护的MNI坐标数量: {len(mni_coordinates)}")
    
    # 创建保护mask
    protection_mask, protection_centers = create_protection_mask(
        data.shape[:3], affine, mni_coordinates, avoid_radius
    )
    print(f"保护的体素数量: {np.sum(protection_mask)}")
    print(f"保护比例: {np.sum(protection_mask)/np.prod(data.shape[:3])*100:.2f}%")

    if use_brain_mask:
        brain_mask = brain_mask_extraction(data[:,:,:,0], method='morphological')
    else:
        brain_mask = None

    # 复制数据以避免修改原始数据
    noisy_data = data.copy()
    
    # 设置默认噪声参数
    if noise_params is None:
        noise_params = get_default_noise_params(noise_type, data)

    print("noise params:", noise_params)
    
    # 处理指定的volumes
    if len(data.shape) == 4:
        if volumes is None:
            volumes = range(data.shape[3])
        elif isinstance(volumes, int):
            volumes = [volumes]
        
        for vol_idx in tqdm(volumes, desc="processing volumes"):
            noisy_data[:,:,:,vol_idx] = add_noise_to_volume(
                data[:,:,:,vol_idx], 
                protection_mask,
                brain_mask,
                noise_type, 
                noise_params
            )
    else:
        # 3D数据
        print("处理3D数据...")
        noisy_data = add_noise_to_volume(
            data, protection_mask, brain_mask, noise_type, noise_params
        )
    
    # band limited noise [0.01~0.08Hz]
    if use_band_filter:
        noise = noisy_data - data
        filtered_noise = band_limited_filtered_noise(noise)
        noisy_data = data + filtered_noise


    # 保存结果
    if output_file:
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        noisy_img = nib.Nifti1Image(noisy_data, affine, img.header)
        nib.save(noisy_img, output_file)
        print(f"保存噪声数据到: {output_file}")
    
    # 保存mask
    if save_mask:
        mask_file = output_file.replace('.nii', '_protection_mask.nii') if output_file else 'protection_mask.nii.gz'
        mask_img = nib.Nifti1Image(protection_mask.astype(np.uint8), affine, img.header)
        nib.save(mask_img, mask_file)
        print(f"保存保护mask到: {mask_file}")
    
    return noisy_data, protection_mask

def brain_mask_extraction(volume_data, method='morphological'):
    """
    高级大脑提取方法
    """
    
    from skimage import morphology, measure
    from skimage.filters import threshold_otsu
    from scipy import ndimage
    
    if method == 'morphological':
        # 形态学方法
        
        # 1. 初始阈值
        threshold = threshold_otsu(volume_data[volume_data > 0])
        binary_mask = volume_data > threshold
        
        # 2. 形态学操作
        # 闭运算填充小孔
        binary_mask = morphology.binary_closing(binary_mask, morphology.ball(3))
        
        # 开运算去除小噪声
        binary_mask = morphology.binary_opening(binary_mask, morphology.ball(2))
        
        # 3. 连通组件分析，保留最大组件
        labeled = measure.label(binary_mask)
        props = measure.regionprops(labeled)
        
        if props:
            # 找到最大的连通组件（通常是大脑）
            largest_label = max(props, key=lambda x: x.area).label
            brain_mask = labeled == largest_label
        else:
            brain_mask = binary_mask
            
    elif method == 'watershed':
        # 分水岭方法
        from skimage.segmentation import watershed
        from skimage.feature import peak_local_maxima
        
        # 使用分水岭进行更精确的分割
        # （这里是简化版，实际应用可能需要更复杂的预处理）
        
        # 距离变换
        threshold = np.percentile(volume_data[volume_data > 0], 30)
        binary = volume_data > threshold
        distance = ndimage.distance_transform_edt(binary)
        
        # 找局部最大值作为种子点
        coords = peak_local_maxima(distance, min_distance=20, threshold_abs=10)
        mask = np.zeros(distance.shape, dtype=bool)
        mask[tuple(coords.T)] = True
        markers, _ = ndimage.label(mask)
        
        # 分水岭分割
        labels = watershed(-distance, markers, mask=binary)
        
        # 选择最大的区域作为大脑
        unique, counts = np.unique(labels[labels > 0], return_counts=True)
        if len(unique) > 0:
            largest_label = unique[np.argmax(counts)]
            brain_mask = labels == largest_label
        else:
            brain_mask = binary
    
    return brain_mask

def create_protection_mask(shape_3d, affine, mni_coordinates, avoid_radius):
    """修正的保护区域mask创建"""
    
    mask = np.zeros(shape_3d, dtype=bool)
    
    # 计算体素大小
    voxel_size = np.abs(affine[0, 0])  # 通常是3mm
    voxel_radius = int(np.ceil(avoid_radius / voxel_size))
    
    print(f"体素大小: {voxel_size:.2f}mm")
    print(f"避开半径: {avoid_radius}mm = {voxel_radius}个体素")
    
    protection_centers = []  # 记录保护中心用于验证
    
    for mni_coord in mni_coordinates:
        # MNI到体素坐标
        mni_homog = np.array([mni_coord[0], mni_coord[1], mni_coord[2], 1])
        voxel_coord = np.linalg.inv(affine) @ mni_homog
        
        # 不要取整，保持精确坐标
        i_center, j_center, k_center = voxel_coord[:3]
        i, j, k = np.round(voxel_coord[:3]).astype(int)
        
        # print(f"MNI{mni_coord} -> 精确体素({i_center:.2f},{j_center:.2f},{k_center:.2f}) -> 整数体素({i},{j},{k})")
        
        # 检查坐标是否在范围内
        if (0 <= i < shape_3d[0] and 
            0 <= j < shape_3d[1] and 
            0 <= k < shape_3d[2]):
            
            # 使用精确中心创建球形保护区域
            mask = add_spherical_protection(mask, (i_center, j_center, k_center), voxel_radius)
            protection_centers.append((i_center, j_center, k_center))
        else:
            print(f"警告: MNI{mni_coord} -> 体素({i},{j},{k}) 超出范围")
    
    return mask, protection_centers

def add_spherical_protection(mask, center, radius):
    """使用精确中心创建球形保护区域"""
    
    i0, j0, k0 = center
    shape = mask.shape
    
    # 扩展搜索范围以确保完整球形
    i_min = max(0, int(np.floor(i0 - radius)))
    i_max = min(shape[0], int(np.ceil(i0 + radius)) + 1)
    j_min = max(0, int(np.floor(j0 - radius)))
    j_max = min(shape[1], int(np.ceil(j0 + radius)) + 1)
    k_min = max(0, int(np.floor(k0 - radius)))
    k_max = min(shape[2], int(np.ceil(k0 + radius)) + 1)
    
    for i in range(i_min, i_max):
        for j in range(j_min, j_max):
            for k in range(k_min, k_max):
                # 计算到精确中心的距离
                distance = np.sqrt((i-i0)**2 + (j-j0)**2 + (k-k0)**2)
                if distance <= radius:
                    mask[i, j, k] = True
    
    return mask

def get_default_noise_params(noise_type, data):
    """获取默认噪声参数"""
    
    data_std = np.std(data[data != 0])  # 只考虑非零体素
    data_mean = np.mean(data[data != 0])
    
    if noise_type == 'gaussian':
        return {
            'mean': 0,
            'std': data_std * 0.1  # 10%的标准差
        }
    elif noise_type == 'uniform':
        return {
            'low': -data_std * 0.1,
            'high': data_std * 0.1
        }
    elif noise_type == 'salt_pepper':
        return {
            'salt_prob': 0.01,
            'pepper_prob': 0.01,
            'salt_value': data_mean + 3*data_std,
            'pepper_value': 0
        }
    elif noise_type == 'multiplicative':
        return {
            'factor_std': 0.1  # 10%的乘性噪声
        }
    else:
        return {}

def add_noise_to_volume(volume, protection_mask, brain_mask=None, noise_type='gaussian', noise_params=None):
    """给单个3D volume添加噪声"""
    # 获取原始数据的有效范围
    data_min = np.min(volume)
    data_max = np.max(volume)
    
    # 对于MRI数据，通常最小值应该>=0
    safe_min = max(0.0, data_min)
    safe_max = data_max

    noisy_volume = volume.copy()
    
    # 创建噪声添加区域的mask（不被保护的区域）
    noise_mask = ~protection_mask
    if brain_mask is not None:
        noise_mask = np.logical_and(brain_mask, ~protection_mask)

    if noise_type == 'gaussian':
        noise = np.random.normal(
            noise_params['mean'], 
            noise_params['std'], 
            volume.shape
        )
        noisy_volume[noise_mask] += noise[noise_mask]
    
    elif noise_type == 'uniform':
        noise = np.random.uniform(
            noise_params['low'], 
            noise_params['high'], 
            volume.shape
        )
        noisy_volume[noise_mask] += noise[noise_mask]
    
    elif noise_type == 'salt_pepper':
        # 椒盐噪声
        salt_mask = (np.random.random(volume.shape) < noise_params['salt_prob']) & noise_mask
        pepper_mask = (np.random.random(volume.shape) < noise_params['pepper_prob']) & noise_mask
        
        noisy_volume[salt_mask] = noise_params['salt_value']
        noisy_volume[pepper_mask] = noise_params['pepper_value']
    
    elif noise_type == 'multiplicative':
        # 乘性噪声
        noise_factor = np.random.normal(1.0, noise_params['factor_std'], volume.shape)
        noisy_volume[noise_mask] *= noise_factor[noise_mask]
    
    # 限制数值范围
    noisy_volume = np.clip(noisy_volume, safe_min, safe_max)

    return noisy_volume


def band_limited_filtered_noise(noise, TR=2.0, low_freq=0.01, high_freq=0.08):
    # nx, ny, nz, nt = noise.shape
    fs = 1 / TR # 采样频率=1/重复时间 = 0.5Hz
    nyquist = fs / 2 # 奈奎斯特频率=0.25Hz
    low_norm = low_freq / nyquist
    high_norm = high_freq / nyquist
    
    if high_norm >= 1.0:
        # 上界无效，只使用高通滤波器，高于low_norm即通过
        b, a = butter(4, low_norm, btype='high')
    else:
        # 使用 巴特沃斯带通滤波器
        b, a = butter(4, [low_norm, high_norm], btype='band')
    
    # 对每个体素的时间序列进行滤波
    filtered_noise = np.zeros_like(noise)
    # for i in range(nx):
    #     for j in range(ny):
    #         for k in range(nz):
    #             filtered_noise[i, j, k, :] = filtfilt(b, a, noise[i, j, k, :])
    filtered_noise = filtfilt(b, a, noise, axis=3)
    return filtered_noise


def test_main():
    input_nii_fpath = "/mnt/c/Works/ws/shoufa2025/data/nii_data_2507/pre_sub_0003F.nii"
    output_nii_fpath = "/mnt/c/Works/ws/shoufa2025/data/nii_data_2507_noised/pre_sub_0003F.nii"
    # 添加高斯噪声，避开指定坐标6mm半径
    noisy_data, mask = add_noise_avoid_coordinates(
        nifti_file=input_nii_fpath,
        mni_coordinates=mni_coords,
        output_file=output_nii_fpath,
        avoid_radius=6,
        noise_type='gaussian',
        noise_params=None,  # {'mean': 0, 'std': 50},
        save_mask=False  # if save protection mask
    )

    print(f"处理完成！噪声数据形状: {noisy_data.shape}")


if __name__ == "__main__":
    input_nii_fpath = "/mnt/c/Works/ws/shoufa2025/data/nii_data_2507/pre_sub_0003F.nii"
    output_nii_fpath = "/mnt/c/Works/ws/shoufa2025/data/nii_data_2507_noised/pre_sub_0003F.nii"
    output_nii_fpath_bandfilt = "/mnt/c/Works/ws/shoufa2025/data/nii_data_2507_noised/pre_sub_0003F_bandfilt.nii"
    # # 添加高斯噪声，避开指定坐标6mm半径
    # noisy_data, mask = add_noise_avoid_coordinates(
    #     nifti_file=input_nii_fpath,
    #     mni_coordinates=mni_coords,
    #     output_file=output_nii_fpath,
    #     avoid_radius=6,
    #     noise_type='gaussian',
    #     noise_params=None,  # {'mean': 0, 'std': 50},
    #     save_mask=False  # if save protection mask
    # )

    # print(f"处理完成！噪声数据形状: {noisy_data.shape}")

    import matplotlib.pyplot as plt
    from scipy import signal
    from scipy.fft import fft, fftfreq
    
    def analyze_power_spectrum(data, TR=2.0):
        """分析功率谱"""
        from scipy import signal
        import numpy as np
        import warnings
        
        # 数据验证
        if len(data.shape) != 4:
            raise ValueError(f"需要4D fMRI数据,得到{len(data.shape)}D")
        
        nx, ny, nz, nt = data.shape
        fs = 1/TR
        
        print(f"fMRI数据功率谱分析:")
        print(f"  数据形状: {data.shape}")
        print(f"  时间点数: {nt}")
        print(f"  TR: {TR}s")
        print(f"  采样频率: {fs:.4f} Hz")
        
        # 选择合适的nperseg，避免警告, Number of data Points PER SEGment
        nperseg = min(64, nt//4) if nt > 16 else nt
        
        # 抑制scipy的警告
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", "nperseg.*greater than input length")
            
            # 正确使用axis=3分析时间维度
            frequencies, psd = signal.welch(data, fs, axis=3, nperseg=nperseg)
        
        # 计算全脑平均功率谱
        mean_psd = np.mean(psd, axis=(0, 1, 2))
        
        # 分析不同频带的能量
        fmri_band = (frequencies >= 0.01) & (frequencies <= 0.08)
        noise_band = frequencies > 0.08
        
        fmri_power = np.mean(mean_psd[fmri_band]) if np.any(fmri_band) else 0
        noise_power = np.mean(mean_psd[noise_band]) if np.any(noise_band) else 0
        
        print(f"  频率分辨率: {frequencies[1]-frequencies[0]:.6f} Hz")
        print(f"  频率范围: {frequencies[0]:.6f} - {frequencies[-1]:.6f} Hz")
        print(f"  fMRI频带功率 (0.01-0.08Hz): {fmri_power:.6f}")
        print(f"  噪声频带功率 (>0.08Hz): {noise_power:.6f}")
        
        if noise_power > 0:
            snr = fmri_power / noise_power
            print(f"  功率比 (信号/噪声): {snr:.3f}")
        else:
            snr = float('inf')
            print(f"  功率比: 无穷大 (无噪声功率)")
        
        return {
            'frequencies': frequencies,
            'psd': psd,  # 完整的4D PSD
            'mean_psd': mean_psd,  # 全脑平均PSD
            'fmri_power': fmri_power,
            'noise_power': noise_power,
            'snr': snr
        }

    def generate_fmri_compatible_noise(shape, TR=2.0, noise_type='band_limited'):
        """生成符合fMRI特性的噪声"""
        
        if len(shape) == 4:  # 4D fMRI数据
            nx, ny, nz, nt = shape
        else:
            raise ValueError("需要4D数据 (x, y, z, time)")
        
        fs = 1/TR # 1/重复时间（秒） = 采样频率 = 0.5 Hz
        
        if noise_type == 'band_limited':
            # 方法1：带限白噪声 (限制在fMRI频带内)
            noise = generate_band_limited_noise(shape, fs, low_freq=0.01, high_freq=0.08)
            
        elif noise_type == 'pink_noise':
            # 方法2：粉红噪声 (1/f噪声，更接近生理噪声)
            noise = generate_pink_noise(shape, fs)
            
        elif noise_type == 'low_frequency':
            # 方法4：低频噪声
            noise = generate_low_frequency_noise(shape, fs, cutoff=0.08)
            
        else:
            raise ValueError(f"不支持的噪声类型: {noise_type}")
        
        return noise

    def generate_band_limited_noise(shape, fs, low_freq=0.01, high_freq=0.08):
        """生成带限白噪声"""
        from scipy.signal import butter, filtfilt
        
        nx, ny, nz, nt = shape
        
        # 生成白噪声
        white_noise = np.random.randn(nx, ny, nz, nt) # standard normal distribution (mu=0,std=1)
        # gen normal distribution with input params {mean,std}
        
        # 设计带通滤波器
        nyquist = fs / 2  # 奈奎斯特频率 = 0.25 Hz
        low_norm = low_freq / nyquist
        high_norm = high_freq / nyquist
        
        if high_norm >= 1.0:
            # 上界无效，只使用高通滤波器，高于low_norm的都通过
            b, a = butter(4, low_norm, btype='high')
        else:
            # 使用 巴特沃斯带通滤波器
            b, a = butter(4, [low_norm, high_norm], btype='band')
        
        # 对每个体素的时间序列进行滤波
        filtered_noise = np.zeros_like(white_noise)
        for i in range(nx):
            for j in range(ny):
                for k in range(nz):
                    filtered_noise[i, j, k, :] = filtfilt(b, a, white_noise[i, j, k, :])
        
        return filtered_noise

    def generate_pink_noise(shape, fs, alpha=1.0):
        """生成粉红噪声 (1/f^alpha)"""
        nx, ny, nz, nt = shape
        
        # 在频域生成粉红噪声
        freqs = fftfreq(nt, 1/fs)
        freqs[0] = 1e-10  # 避免除零
        
        # 1/f^alpha 功率谱
        power_spectrum = 1 / (np.abs(freqs) ** alpha)
        power_spectrum[0] = 0  # DC分量为0
        
        pink_noise = np.zeros((nx, ny, nz, nt))
        
        for i in range(nx):
            for j in range(ny):
                for k in range(nz):
                    # 生成随机相位
                    phases = np.random.uniform(0, 2*np.pi, nt)
                    # 构造复数谱
                    spectrum = np.sqrt(power_spectrum) * np.exp(1j * phases)
                    # 逆傅里叶变换得到时域信号
                    pink_noise[i, j, k, :] = np.real(np.fft.ifft(spectrum))
        
        return pink_noise

    def generate_low_frequency_noise(shape, fs, cutoff=0.08):
        """生成低频噪声"""
        from scipy.signal import butter, filtfilt
        
        nx, ny, nz, nt = shape
        
        # 生成白噪声
        white_noise = np.random.randn(nx, ny, nz, nt)
        
        # 低通滤波器
        nyquist = fs / 2
        normalized_cutoff = cutoff / nyquist
        b, a = butter(4, normalized_cutoff, btype='low')
        
        # 滤波
        filtered_noise = np.zeros_like(white_noise)
        for i in range(nx):
            for j in range(ny):
                for k in range(nz):
                    filtered_noise[i, j, k, :] = filtfilt(b, a, white_noise[i, j, k, :])
        
        return filtered_noise

    # TODO: check new band_limited_noise() method, replace former gaussian noise
    # 添加高斯噪声，避开指定坐标6mm半径
    noisy_data, mask = add_noise_avoid_coordinates(
        nifti_file=input_nii_fpath,
        mni_coordinates=mni_coords,
        output_file=output_nii_fpath_bandfilt,
        avoid_radius=6,
        noise_type='gaussian',
        noise_params=None,  # {'mean': 0, 'std': 50},
        save_mask=False,  # if save protection mask
        use_brain_mask=False
    )

    # analyze signal power
    img = nib.load(output_nii_fpath_bandfilt)  # input_nii_fpath   output_nii_fpath output_nii_fpath_bandfilt
    data = img.get_fdata()
    affine = img.affine
    analyze_power_spectrum(data, TR=2.0)
