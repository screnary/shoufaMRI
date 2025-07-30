import os

import utils.dir_utils as dir_utils
import utils.add_noise_to_volumes as noise
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


def main_1():
    # 处理硬盘内的数据
    # data_root = "/mnt/c/Works/ws/shoufa2025/data"
    data_root = "/mnt/e/05_original_data_no_editing_processed"  # 移动硬盘
    nii_dir = os.path.join(data_root, 'pre_surgery', 'RestTARWSDCF')
    nii_dir_2 = os.path.join(data_root, 'post_surgery', 'RestTARWSDCF')
    print(f"processing path {nii_dir}")
    nii_fnames = dir_utils.get_nii_files_with_pattern(nii_dir)
    nii_fnames_2 = dir_utils.get_nii_files_with_pattern(nii_dir_2)
    nii_fnames.extend(nii_fnames_2)
    total_fcount = len(nii_fnames)

    # 添加高斯噪声，避开指定坐标6mm半径
    for i,nii_fn in enumerate(nii_fnames):
        print(f"********** Processing {i+1}/{total_fcount}")
        input_nii_fpath = nii_fn  # '/mnt/e/05_original_data_no_editing_processed/pre_surgery/RestTARWSDCF/sub_0001/pre_sub_0001F.nii'
        output_nii_fpath = input_nii_fpath.replace('/05_original_data_no_editing_processed/', '/05_original_data_no_editing_processed_addnoise/')

        noisy_data, mask = noise.add_noise_avoid_coordinates(
            nifti_file=input_nii_fpath,
            mni_coordinates=mni_coords,
            output_file=output_nii_fpath,
            avoid_radius=6,
            noise_type='gaussian',
            noise_params=None,  # {'mean': 0, 'std': 50},
            save_mask=False,  # if save protection mask
            use_brain_mask=True
        )

def main_2():
    # 处理本地数据，为了进行可视化 demo
    data_root = "/mnt/c/Works/ws/shoufa2025/data"  # local data
    nii_dir = os.path.join(data_root, 'nii_data_2507')
    print(f"processing path {nii_dir}")
    nii_fnames = dir_utils.get_nii_files_with_pattern(nii_dir)
    total_fcount = len(nii_fnames)

    # 添加高斯噪声，避开指定坐标6mm半径
    for i,nii_fn in enumerate(nii_fnames):
        print(f"********** Processing {i+1}/{total_fcount}")
        input_nii_fpath = nii_fn
        output_nii_fpath = input_nii_fpath.replace('/nii_data_2507/', '/nii_data_2507_noised_new/')

        noisy_data, mask = noise.add_noise_avoid_coordinates(
            nifti_file=input_nii_fpath,
            mni_coordinates=mni_coords,
            output_file=output_nii_fpath,
            avoid_radius=6,
            noise_type='gaussian',
            noise_params=None,  # {'mean': 0, 'std': 50},
            save_mask=False,  # if save protection mask
            use_brain_mask=True
        )


if __name__ == "__main__":
    # 实验：分组批量处理实验，探索noise的幅度、protection region大小的影响
    print("Please Using Experiment Framework!")
