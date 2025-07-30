'''
202507
Add noise to nifti volumes, avoiding coordinates(MNI space)
'''
import nibabel as nib
import numpy as np
from scipy.ndimage import binary_dilation
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
                               use_brain_mask=False):
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
    protection_mask = create_protection_mask(
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
    voxel_sizes = np.abs(np.diag(affine)[:3])
    voxel_size = np.mean(voxel_sizes)  # 使用平均体素大小
    voxel_radius = int(np.ceil(avoid_radius / voxel_size))
    
    print(f"体素大小: {voxel_sizes}")
    print(f"平均体素大小: {voxel_size:.2f}mm")
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


"""
>>>>>>>>>>>>>>>>>>>>>>>不再使用: 坐标对齐不精确
"""
# def create_protection_mask(shape_3d, affine, mni_coordinates, avoid_radius):
#     """创建保护区域mask"""
    
#     mask = np.zeros(shape_3d, dtype=bool)
    
#     # 计算体素大小（假设各向同性）
#     voxel_size = np.abs(affine[0, 0])  # 通常是3mm
#     voxel_radius = int(np.ceil(avoid_radius / voxel_size))
    
#     print(f"体素大小: {voxel_size}mm")
#     print(f"避开半径: {avoid_radius}mm = {voxel_radius}个体素")
    
#     for mni_coord in mni_coordinates:
#         # 将MNI坐标转换为体素坐标
#         mni_homog = np.array([mni_coord[0], mni_coord[1], mni_coord[2], 1])
#         voxel_coord = np.linalg.inv(affine) @ mni_homog
#         i, j, k = np.round(voxel_coord[:3]).astype(int)
        
#         # 检查坐标是否在范围内
#         if (0 <= i < shape_3d[0] and 
#             0 <= j < shape_3d[1] and 
#             0 <= k < shape_3d[2]):
            
#             # 创建球形保护区域
#             mask = add_spherical_protection(mask, (i, j, k), voxel_radius)
#             # print(f"MNI{mni_coord} -> 体素({i},{j},{k}) 已保护")
#         else:
#             print(f"警告: MNI{mni_coord} -> 体素({i},{j},{k}) 超出范围")
    
#     return mask

# def add_spherical_protection(mask, center, radius):
#     """在mask中添加球形保护区域"""
    
#     i0, j0, k0 = center
#     shape = mask.shape
    
#     # 创建网格
#     i_range = range(max(0, i0-radius), min(shape[0], i0+radius+1))
#     j_range = range(max(0, j0-radius), min(shape[1], j0+radius+1))
#     k_range = range(max(0, k0-radius), min(shape[2], k0+radius+1))
    
#     for i in i_range:
#         for j in j_range:
#             for k in k_range:
#                 # 计算到中心的距离
#                 distance = np.sqrt((i-i0)**2 + (j-j0)**2 + (k-k0)**2)
#                 if distance <= radius:
#                     mask[i, j, k] = True
    
#     return mask
"""
<<<<<<<<<<<<<<<<<<<<<<<<<<<不再使用: 坐标对齐不精确
"""

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

def visualize_protection_regions(nifti_file, mni_coordinates, avoid_radius=5, slice_coords=None):
    """可视化保护区域"""
    
    import matplotlib.pyplot as plt
    
    img = nib.load(nifti_file)
    data = img.get_fdata()
    affine = img.affine
    
    # 创建保护mask
    mask = create_protection_mask(data.shape[:3], affine, mni_coordinates, avoid_radius)
    
    # 选择显示的切片
    if slice_coords is None:
        slice_coords = [s//2 for s in data.shape[:3]]  # 中心切片
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # 如果是4D数据，取第一个volume
    if len(data.shape) == 4:
        display_data = data[:,:,:,0]
    else:
        display_data = data
    
    # 轴状面 (Z切片) 与地面平行，将人体分为上下两部分
    axes[0].imshow(display_data[:,:,slice_coords[2]].T, cmap='gray', origin='lower')
    axes[0].imshow(mask[:,:,slice_coords[2]].T, cmap='Reds', alpha=0.5, origin='lower')
    axes[0].set_title(f'Axial plane (Z={slice_coords[2]})')
    axes[0].set_xlabel('X (Left-Right)')
    axes[0].set_ylabel('Y (Rare-Front)')
    
    # 冠状面 (Y切片) 与前额平行，将人体分为前（腹侧）后（背侧）两部分
    axes[1].imshow(display_data[:,slice_coords[1],:].T, cmap='gray', origin='lower')
    axes[1].imshow(mask[:,slice_coords[1],:].T, cmap='Reds', alpha=0.5, origin='lower')
    axes[1].set_title(f'Coronal plane (Y={slice_coords[1]})')
    axes[1].set_xlabel('X (Left-Right)')
    axes[1].set_ylabel('Z (Bottom-Top)')
    
    # 矢状面 (X切片) 从前到后将人体分为左右两部分
    axes[2].imshow(display_data[slice_coords[0],:,:].T, cmap='gray', origin='lower')
    axes[2].imshow(mask[slice_coords[0],:,:].T, cmap='Reds', alpha=0.5, origin='lower')
    axes[2].set_title(f'Sagittal plane (X={slice_coords[0]})')
    axes[2].set_xlabel('Y (Rare-Front)')
    axes[2].set_ylabel('Z (Bottom-Top)')
    
    plt.tight_layout()
    plt.suptitle(f'Protection Region (Red=Protedted, Radius={avoid_radius}mm)', y=1.02)
    plt.show()
    
    # 显示保护的MNI坐标
    print("保护的MNI坐标:")
    for i, coord in enumerate(mni_coordinates):
        print(f"  {i+1}. MNI{coord}")

# analyse noise amplify
def compare_noise_effects(nifti_file, mni_coordinates, avoid_radius=6, output_file=None):
    """比较不同噪声类型的效果"""
    
    # 先加载原始数据检查维度
    original_img = nib.load(nifti_file)
    original_full = original_img.get_fdata()
    is_4d = len(original_full.shape) == 4
    
    print(f"原始数据形状: {original_full.shape}")
    
    noise_types = ['gaussian', 'uniform', 'salt_pepper', 'multiplicative']
    
    for noise_type in noise_types:
        # output_file = f'noisy_{noise_type}.nii.gz'
        
        # 对于4D数据，只处理第一个volume进行比较
        volumes_to_process = 0 if is_4d else None
        
        noisy_data, mask = add_noise_avoid_coordinates(
            nifti_file=nifti_file,
            mni_coordinates=mni_coordinates,
            output_file=output_file,
            avoid_radius=avoid_radius,
            noise_type=noise_type,
            volumes=volumes_to_process
        )
        
        print(f"\n{noise_type.upper()} 噪声统计:")
        
        # 确保维度匹配
        if is_4d:
            # 取第一个volume进行比较
            original = original_full[:,:,:,0]
            if len(noisy_data.shape) == 4:
                noisy_volume = noisy_data[:,:,:,0]
            else:
                noisy_volume = noisy_data
        else:
            original = original_full
            noisy_volume = noisy_data
        
        print(f"  原始数据形状: {original.shape}")
        print(f"  噪声数据形状: {noisy_volume.shape}")
        
        # 计算差异
        diff = noisy_volume - original
        
        # 分析噪声区域和保护区域
        noise_region = diff[~mask]
        protected_region = diff[mask]
        
        print(f"  噪声区域均值: {np.mean(noise_region):.3f}")
        print(f"  噪声区域标准差: {np.std(noise_region):.3f}")
        print(f"  噪声区域范围: [{np.min(noise_region):.3f}, {np.max(noise_region):.3f}]")
        print(f"  保护区域最大变化: {np.max(np.abs(protected_region)):.6f} (应该接近0)")
        print(f"  保护区域平均变化: {np.mean(np.abs(protected_region)):.6f}")


if __name__ == "__main__":
    input_nii_fpath = "/mnt/c/Works/ws/shoufa2025/data/nii_data_2507/pre_sub_0003F.nii"
    output_nii_fpath = "/mnt/c/Works/ws/shoufa2025/data/nii_data_2507_noised/pre_sub_0003F.nii"
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

    # # Simple 可视化 protection mask (select centre slices)
    # visualize_protection_regions(input_nii_fpath, mni_coords, avoid_radius=6)

    # analyse noise effect
    compare_noise_effects(input_nii_fpath, mni_coords, avoid_radius=6)
