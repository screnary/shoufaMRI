import pyvista as pv
import nibabel as nib
import numpy as np
import pdb


def visualize_volume_3d(nifti_file, volume_idx=0, visualization_type='volume', 
                       threshold=None, opacity=0.3, cmap='viridis'):
    """
    使用PyVista进行NIfTI volume的3D可视化（更新版本）
    
    参数:
    - nifti_file: NIfTI文件路径
    - volume_idx: volume索引（对4D数据）
    - visualization_type: 可视化类型 ('volume', 'isosurface', 'contour', 'slices')
    - threshold: 阈值（用于等值面）
    - opacity: 透明度
    - cmap: 颜色映射
    """
    
    print(f"=== 开始3D可视化: {nifti_file} ===")
    
    try:
        # 加载数据
        img = nib.load(nifti_file)
        data = img.get_fdata()
        affine = img.affine
        
        print(f"数据形状: {data.shape}")
        print(f"数据类型: {data.dtype}")
        
        # 选择volume
        if len(data.shape) == 4:
            if volume_idx >= data.shape[3]:
                print(f"错误: volume索引 {volume_idx} 超出范围 (0-{data.shape[3]-1})")
                return None, None
            volume_data = data[:, :, :, volume_idx]
            print(f"选择Volume {volume_idx}")
        else:
            volume_data = data
            print("3D数据")
        
        # 数据预处理
        volume_data = preprocess_volume_data(volume_data)
        
        # 创建PyVista的StructuredGrid
        grid = create_pyvista_grid(volume_data, affine)
        
        # 数据统计
        print_data_statistics(volume_data)
        
        # 创建绘图器
        plotter = pv.Plotter(window_size=(1200, 800))
        plotter.set_background('white')
        
        # 根据可视化类型进行渲染
        success = False
        
        if visualization_type == 'volume':
            # 体渲染（使用新的函数）
            success = visualize_volume_rendering(plotter, grid, opacity, cmap)
            
        elif visualization_type == 'isosurface':
            # 等值面渲染
            if threshold is None:
                threshold = calculate_auto_threshold(volume_data)
            success = visualize_isosurface(plotter, grid, threshold, cmap)
            
        elif visualization_type == 'contour':
            # 等高线渲染
            success = visualize_contours(plotter, grid, cmap)
            
        elif visualization_type == 'slices':
            # 切片渲染
            success = visualize_slices(plotter, grid, cmap)
            
        else:
            print(f"未知的可视化类型: {visualization_type}")
            print("可用类型: 'auto', 'volume', 'isosurface', 'contour', 'slices'")
            return None, None
        
        if not success:
            print("所有可视化方法都失败，使用基本切片...")
            visualize_basic_slices(plotter, grid, cmap)
        
        # 添加坐标轴和标题
        plotter.add_axes()
        plotter.add_title(f"Volume {volume_idx} - {visualization_type.capitalize()}", 
                         font_size=16)
        
        # 添加颜色条
        plotter.add_scalar_bar(title="强度值")

        # 显示
        plotter.show()
        
        return grid, plotter
        
    except Exception as e:
        print(f"可视化失败: {e}")
        import traceback
        traceback.print_exc()
        return None, None

def preprocess_volume_data(volume_data):
    """数据预处理"""
    
    print("预处理数据...")
    
    # 处理NaN和无穷值
    volume_data = np.nan_to_num(volume_data, nan=0.0, posinf=0.0, neginf=0.0)
    
    # 数据范围限制（去除极值）
    if np.any(volume_data != 0):
        non_zero_data = volume_data[volume_data != 0]
        p_bot, p_top = np.percentile(non_zero_data, [5, 95])  #[1, 99]
        volume_data = np.clip(volume_data, p_bot, p_top)
    
    print(f"预处理后数据范围: {volume_data.min():.2f} - {volume_data.max():.2f}")
    
    return volume_data

def print_data_statistics(volume_data):
    """打印数据统计信息"""
    
    print("=== 数据统计 ===")
    print(f"形状: {volume_data.shape}")
    print(f"数据范围: {volume_data.min():.2f} - {volume_data.max():.2f}")
    print(f"平均值: {volume_data.mean():.2f}")
    print(f"标准差: {volume_data.std():.2f}")
    
    non_zero_count = np.count_nonzero(volume_data)
    total_count = volume_data.size
    print(f"非零值: {non_zero_count}/{total_count} ({100*non_zero_count/total_count:.1f}%)")

def calculate_auto_threshold(volume_data):
    """自动计算合适的阈值"""
    
    non_zero_data = volume_data[volume_data > 0]
    
    if len(non_zero_data) == 0:
        return volume_data.max() * 0.5
    
    # 使用多种方法计算阈值，选择最合适的
    thresholds = {
        'mean': non_zero_data.mean(),
        'median': np.median(non_zero_data),
        '75th_percentile': np.percentile(non_zero_data, 75),
        'otsu': calculate_otsu_threshold(non_zero_data)
    }
    
    print("自动阈值计算:")
    for name, value in thresholds.items():
        print(f"  {name}: {value:.2f}")
    
    # 选择75th百分位数作为默认
    selected_threshold = thresholds['75th_percentile']
    print(f"选择阈值: {selected_threshold:.2f}")
    
    return selected_threshold

def calculate_otsu_threshold(data):
    """计算Otsu阈值"""
    
    try:
        from skimage.filters import threshold_otsu
        return threshold_otsu(data)
    except ImportError:
        # 如果没有skimage，使用简单的方法
        hist, bin_edges = np.histogram(data, bins=50)
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
        
        # 简单的双峰检测
        max_idx = np.argmax(hist)
        valley_idx = max_idx + np.argmin(hist[max_idx:])
        
        return bin_centers[valley_idx] if valley_idx < len(bin_centers) else data.mean()

def create_pyvista_grid(volume_data, affine):
    """创建PyVista的结构化网格（更新版本）"""
    
    print("创建PyVista网格...")
    
    # 获取体素间距
    voxel_size = np.abs(np.diag(affine)[:3])
    
    # 创建坐标
    if len(volume_data.shape) == 4:
        data = volume_data[:, :, :, 0]
        print(f"选择Volume {0}")
    else:
        data = volume_data
        print("3D数据")

    nx, ny, nz = data.shape
    
    # 使用ImageData更高效
    grid = pv.ImageData(dimensions=(nx, ny, nz), spacing=voxel_size)
    
    # 添加数据
    grid.point_data['values'] = data.ravel()  # flattening to 1d array
    
    print(f"网格创建完成: {grid.n_points} 点, {grid.n_cells} 单元")
    
    return grid

# 必要的可视化函数
def visualize_volume_rendering(plotter, grid, opacity=0.1, cmap='viridis'):
    # 数据预处理
    values = grid.point_data['values']
    print(f"数据范围: {values.min():.2f} - {values.max():.2f}")
    print(f"数据点数: {len(values)}")
    
    # 计算有意义的数据阈值
    non_zero_values = values[values > 0]
    
    if len(non_zero_values) == 0:
        print("警告: 没有正值数据")
        return False
    
    # 使用百分位数确定阈值
    threshold_low = np.fmax(0.0, np.percentile(non_zero_values, 25))  # 排除最低10%
    threshold_high = np.percentile(non_zero_values, 95)  # 排除最高5%

    
    print(f"使用阈值范围: {threshold_low:.2f} - {threshold_high:.2f}")
    
    # 方法选择
    methods = [
        ("体渲染", render_volume)
    ]
    
    for method_name, method_func in methods:
        try:
            print(f"尝试{method_name}...")
            success = method_func(plotter, grid, threshold_low, threshold_high, opacity, cmap)
            if success:
                print(f"✓ {method_name}成功")
                return True
        except Exception as e:
            print(f"✗ {method_name}失败: {e}")
            continue
    
    print("所有渲染方法都失败了")
    return False

def render_volume(plotter, grid, threshold_low, threshold_high, opacity, cmap):
    """纯体渲染"""
    
    # 创建阈值过滤的网格
    filtered_grid = grid.threshold([threshold_low, threshold_high])
    
    if filtered_grid.n_points == 0:
        return False
    
    # 体渲染
    plotter.add_volume(
        filtered_grid,
        scalars='values',
        opacity=min(opacity, 0.6),  # 限制opacity
        cmap=cmap,
        show_scalar_bar=True
    )
    
    return True

def visualize_isosurface(plotter, grid, threshold, cmap):
    """等值面可视化"""
    
    print(f"创建等值面，阈值: {threshold}")
    
    try:
        contours = grid.contour([threshold])
        
        if contours.n_points > 0:
            plotter.add_mesh(
                contours,
                cmap=cmap,
                opacity=0.8,
                show_edges=False,
                smooth_shading=True
            )
            print(f"✓ 等值面创建成功: {contours.n_points} 点")
            return True
        else:
            print("✗ 等值面无数据点")
            return False
            
    except Exception as e:
        print(f"✗ 等值面创建失败: {e}")
        return False

def visualize_contours(plotter, grid, cmap):
    """多等高线可视化"""
    
    print("创建多等高线...")
    
    try:
        data_range = grid.get_data_range()
        
        # 创建多个等高线级别
        levels = np.linspace(
            data_range[0] + 0.2 * (data_range[1] - data_range[0]),
            data_range[1] - 0.1 * (data_range[1] - data_range[0]),
            5
        )
        
        contours = grid.contour(levels)
        
        if contours.n_points > 0:
            plotter.add_mesh(
                contours,
                cmap=cmap,
                opacity=0.7,
                show_edges=False
            )
            print(f"✓ 等高线创建成功: {len(levels)} 层级")
            return True
        else:
            print("✗ 等高线无数据点")
            return False
            
    except Exception as e:
        print(f"✗ 等高线创建失败: {e}")
        return False

def visualize_slices(plotter, grid, cmap):
    """切片可视化"""
    
    print("创建正交切片...")
    
    try:
        # 过滤低值
        data_range = grid.get_data_range()
        threshold = data_range[0] + 0.1 * (data_range[1] - data_range[0])
        
        filtered_grid = grid.threshold(threshold)
        
        if filtered_grid.n_points > 0:
            slices = filtered_grid.slice_orthogonal()  # TODO: 指定具体slice
            plotter.add_mesh(slices, cmap=cmap, opacity=0.8)
            print("✓ 切片创建成功")
            return True
        else:
            # 如果过滤后无数据，使用原始数据
            slices = grid.slice_orthogonal()
            plotter.add_mesh(slices, cmap=cmap, opacity=0.8)
            print("✓ 原始切片创建成功")
            return True
            
    except Exception as e:
        print(f"✗ 切片创建失败: {e}")
        return False

def visualize_basic_slices(plotter, grid, cmap):
    """基本切片可视化（最后的备用方案）"""
    
    print("使用基本切片可视化...")
    
    try:
        # 最简单的中间切片
        dims = grid.dimensions
        
        # X中间切片
        x_slice = grid.slice(normal='x', origin=[dims[0]//2, dims[1]//2, dims[2]//2])
        plotter.add_mesh(x_slice, cmap=cmap, opacity=0.7)
        
        # Y中间切片  
        y_slice = grid.slice(normal='y', origin=[dims[0]//2, dims[1]//2, dims[2]//2])
        plotter.add_mesh(y_slice, cmap=cmap, opacity=0.7)
        
        # Z中间切片
        z_slice = grid.slice(normal='z', origin=[dims[0]//2, dims[1]//2, dims[2]//2])
        plotter.add_mesh(z_slice, cmap=cmap, opacity=0.7)
        
        print("✓ 基本切片创建成功")
        return True
        
    except Exception as e:
        print(f"✗ 基本切片也失败: {e}")
        return False


# using 3D visualization for data analysing
def visualize_noise_comparison_3d(original_file, noisy_file, volume_idx=0, 
                                 mni_coordinates=None, avoid_radius=5):
    """3D可视化噪声前后对比"""
    
    # 加载数据
    original_img = nib.load(original_file)
    noisy_img = nib.load(noisy_file)
    
    original_data = original_img.get_fdata()
    noisy_data = noisy_img.get_fdata()
    
    # 选择volume
    if len(original_data.shape) == 4:
        original_vol = original_data[:, :, :, volume_idx]
        noisy_vol = noisy_data[:, :, :, volume_idx]
    else:
        original_vol = original_data
        noisy_vol = noisy_data
    
    # 计算差异
    diff = noisy_vol - original_vol
    
    # 创建网格
    original_grid = create_pyvista_grid(original_vol, original_img.affine)
    noisy_grid = create_pyvista_grid(noisy_vol, original_img.affine)
    diff_grid = create_pyvista_grid(diff, original_img.affine)
    
    # 创建子图
    plotter = pv.Plotter(shape=(2, 2), window_size=(1600, 1200))
    plotter.set_background('white')
    
    # 原始数据
    plotter.subplot(0, 0)
    plotter.add_title("原始数据")
    threshold_orig = np.percentile(original_vol[original_vol > 0], 70)
    iso_orig = original_grid.contour([threshold_orig])
    if iso_orig.n_points > 0:
        plotter.add_mesh(iso_orig, color='blue', opacity=0.8)
    
    # 噪声数据
    plotter.subplot(0, 1)
    plotter.add_title("添加噪声后")
    threshold_noisy = np.percentile(noisy_vol[noisy_vol > 0], 70)
    iso_noisy = noisy_grid.contour([threshold_noisy])
    if iso_noisy.n_points > 0:
        plotter.add_mesh(iso_noisy, color='red', opacity=0.8)
    
    # 差异可视化
    plotter.subplot(1, 0)
    plotter.add_title("差异 (噪声-原始)")
    # 只显示显著差异
    diff_threshold = np.std(diff[diff != 0]) * 2
    pos_diff = diff_grid.threshold(diff_threshold)
    neg_diff = diff_grid.threshold(-diff_threshold, invert=True)
    
    if pos_diff.n_points > 0:
        plotter.add_mesh(pos_diff, color='red', opacity=0.7)
    if neg_diff.n_points > 0:
        plotter.add_mesh(neg_diff, color='blue', opacity=0.7)
    
    # 保护区域可视化（如果提供了坐标）
    plotter.subplot(1, 1)
    plotter.add_title("保护区域")
    if mni_coordinates is not None:
        visualize_protection_regions_3d(plotter, original_img.affine, 
                                       original_vol.shape, mni_coordinates, 
                                       avoid_radius)
    
    # 添加MNI坐标点
    if mni_coordinates is not None:
        add_mni_coordinates_to_plot(plotter, mni_coordinates, original_img.affine)
    
    plotter.show()

def visualize_protection_regions_3d(plotter, affine, shape, mni_coordinates, avoid_radius):
    """3D可视化保护区域"""
    
    # 创建保护mask
    from utils.add_noise_to_volumes import create_protection_mask  # 使用之前定义的函数
    protection_mask = create_protection_mask(shape, affine, mni_coordinates, avoid_radius)
    
    # 创建保护区域网格
    protection_grid = create_pyvista_grid(protection_mask.astype(float), affine)
    
    # 创建等值面
    iso_protection = protection_grid.contour([0.5])
    
    if iso_protection.n_points > 0:
        plotter.add_mesh(iso_protection, color='yellow', opacity=0.6, 
                        label='保护区域')


def add_mni_coordinates_to_plot(plotter, mni_coordinates, affine):
    """在3D图中添加MNI坐标点"""
    
    # 转换MNI坐标到体素坐标，再到物理坐标
    voxel_size = np.abs(np.diag(affine)[:3])
    
    points = []
    for mni_coord in mni_coordinates:
        # MNI到体素坐标
        mni_homog = np.array([mni_coord[0], mni_coord[1], mni_coord[2], 1])
        voxel_coord = np.linalg.inv(affine) @ mni_homog
        
        # 体素坐标到物理坐标（用于PyVista）
        phys_coord = voxel_coord[:3] * voxel_size
        points.append(phys_coord)
    
    if points:
        points_array = np.array(points)
        point_cloud = pv.PolyData(points_array)
        plotter.add_mesh(point_cloud, color='red', point_size=10, 
                        render_points_as_spheres=True, label='MNI坐标点')


# get 
def get_vis_config(nifti_file, volume_idx=0):
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

    img = nib.load(nifti_file)
    data = img.get_fdata()
    affine = img.affine

    # 选择volume
    if len(data.shape) == 4:
        if volume_idx >= data.shape[3]:
            print(f"错误: volume索引 {volume_idx} 超出范围 (0-{data.shape[3]-1})")
            return None, None
        volume_data = data[:, :, :, volume_idx]
        print(f"选择Volume {volume_idx}")
    else:
        volume_data = data
        print("3D数据")
    
    # 数据预处理
    volume_data = preprocess_volume_data(volume_data)
    
    # 创建PyVista的StructuredGrid
    grid = create_pyvista_grid(volume_data, affine)

    return data.shape, affine, mni_coords, grid
