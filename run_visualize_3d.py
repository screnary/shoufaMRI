"""
visualize the noised volumes, check the protection area mask
"""
import os,sys
import pyvista as pv
import pdb

# 快速路径设置
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)  # code/
sys.path.insert(0, parent_dir)
# 设置包名
if __package__ is None:
    # 根据文件位置动态设置包名
    relative_path = os.path.relpath(current_dir, parent_dir)
    __package__ = relative_path.replace(os.sep, '.')
    print(f"设置包名: {__package__}")  # shoufaMRI
try:
    import shoufaMRI.utils.directory_utils as directory_utils
    import shoufaMRI.core.volume_noise_operations as noise
    import shoufaMRI.utils.visualize as vis3d
except ImportError as e:
    print(f"Import Error!!!:{e}")
    sys.exit(1)

if __name__ == "__main__":

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

    # data_root = "/mnt/c/Works/ws/shoufa2025/data"
    data_root = "C:\\Works\\ws\\shoufa2025\\data"  # 移动硬盘

    # # noise added
    # nii_dir = os.path.join(data_root, '05_original_data_no_editing_processed_addnoise', 'pre_surgery', 'RestTARWSDCF')
    # nii_fnames = dir_utils.get_nii_files_with_pattern(nii_dir)

    # original data
    nii_dir_2 = os.path.join(data_root, 'nii_data_2507_noised_new')
    nii_fnames_2 = directory_utils.get_nii_files_with_pattern(nii_dir_2)
    # noised data
    nii_dir_3 = os.path.join(data_root, 'nii_data_2507')
    nii_fnames_3 = directory_utils.get_nii_files_with_pattern(nii_dir_3)

    nii_fname = nii_fnames_2[0]
    nii_fname_noised = nii_fnames_3[0]
    # _, plotter = vis3d.visualize_volume_3d(nii_fname, volume_idx=0, visualization_type='volume',  # 'volume', 'isosurface', 'contour', 'slices'
    #                    threshold=None, opacity=0.4, cmap='viridis')

    # # check protection regions
    datashape, affine, mni_coords, grid= vis3d.get_vis_config(nii_fname)
    # plotter = pv.Plotter(window_size=(1200, 800))
    # plotter.set_background('white')
    # vis3d.visualize_slices(plotter, grid, cmap='viridis')
    # vis3d.visualize_protection_regions_3d(plotter, affine, datashape, mni_coords, avoid_radius=8)
    # plotter.show()

    vis3d.visualize_noise_comparison_3d(nii_fnames_2[0], nii_fnames_3[0], volume_idx=0, 
                                        mni_coordinates=mni_coords, avoid_radius=6)
