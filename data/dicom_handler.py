""" Expand origin dicom data through random volume expantion,
    Output new dicom files
    before: getNewDicomData.py
"""
import os
import glob
import numpy as np
from collections import defaultdict
import cv2
import matplotlib.pyplot as plt
import pydicom
from pydicom.dataset import Dataset, FileDataset
from tqdm import tqdm
import preprocess_data as pp
from datetime import datetime
import argparse
import shutil
import re

import nibabel as nib
import pdb


proj_root = os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))


def argument_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_root', type=str,
                        default="/mnt/e/data/liuyang/",
                        help='data root for this project')
    parser.add_argument('--surg_time', type=str,
                        default="Post_Surgery", choices=['Post_Surgery','Pre_Surgery']
                        )
    parser.add_argument('--data_phase', type=str,
                        default="Rest", choices=['Rest','Struc']
                        )

    args = parser.parse_args()
    return args


def read_dicom_metadata(dataset):
    # 打印文件元信息（如果存在）
    if hasattr(dataset, 'file_meta'):
        print("File Meta Information:")
        for elem in dataset.file_meta:
            print(f"  {elem.tag}: {elem.name} = {elem.value}")
        print()

    # 打印数据集元素
    print("Dataset Elements:")
    for elem in dataset:
        # 检查是否是序列或嵌套数据集
        if elem.VR == "SQ":
            print(f"  {elem.tag}: {elem.name} = Sequence:")
            for i, item in enumerate(elem):
                print(f"    Item {i}:")
                for subelem in item:
                    print(f"      {subelem.tag}: {subelem.name} = {subelem.value}")
                # pdb.set_trace()
        else:
            # 对于像素数据，只打印标签和名称，不打印值（因为可能非常大）
            if elem.tag == (0x7fe0, 0x0010):  # Pixel Data
                print(f"  {elem.tag}: {elem.name} = [Pixel Data]")
            else:
                print(f"  {elem.tag}: {elem.name} = {elem.value}")


def parseDicomFile(file_path):
    """ read MRI data and parse meta data, pixel data
    input : file_path(str), file name full path 
    return: 
    """
    dataset = pydicom.dcmread(file_path)
    # read common meta data
    read_dicom_metadata(dataset)
    # pdb.set_trace()
    # read unique meta data
    
    # read slice pixel data (matrix)


def update_dicom(input_file, output_file, updates):
    # 读取原始DICOM文件
    original_dcm = pydicom.dcmread(input_file)

    # 创建一个新的 FileDataset 对象，复制原始数据集
    new_dcm = FileDataset(output_file, {}, file_meta=original_dcm.file_meta, preamble=original_dcm.preamble)

    # 复制所有原始元素到新数据集
    for elem in original_dcm:
        new_dcm.add(elem)

    # 更新指定的元数据
    for key, value in updates.items():
        if hasattr(new_dcm, key):
            setattr(new_dcm, key, value)
        else:
            print(f"警告: {key} 不是有效的DICOM元素。")

    # 更新 SOP Instance UID，因为这是一个新的实例
    new_dcm.SOPInstanceUID = pydicom.uid.generate_uid()

    # 设置当前日期和时间
    dt = datetime.now()
    new_dcm.ContentDate = dt.strftime('%Y%m%d')
    new_dcm.ContentTime = dt.strftime('%H%M%S.%f')

    # 保存新的DICOM文件
    new_dcm.save_as(output_file)

    print(f"更新后的DICOM文件已保存为: {output_file}")

''' # 使用示例
input_file = "path/to/your/input.dcm"
output_file = "path/to/your/output.dcm"

# 指定要更新的元数据
updates = {
    'PatientName': 'SMITH^JOHN',
    'PatientID': '12345',
    'StudyDescription': 'Updated Study',
    'SeriesDescription': 'Updated Series'
} '''

# update_dicom(input_file, output_file, updates)

def getVidIdx(volume_size=40, total_num=12000):
    def getBaseIdx(volume_size=40):
        # prepare imlist file (.npz file, numpy list [[0,39,1,38,2,37,...],[],[],...])
        # output: 0,39,1,38,2,37,...;
        idx = list(range(0,40,1))
        idx_inv = idx[::-1]
        idx_base = []
        for i in range(volume_size//2):
            idx_base.append(idx[i])
            idx_base.append(idx_inv[i])
        return np.asarray(idx_base)

    idx_base = getBaseIdx(volume_size)
    vid_list = [] # volume id list

    for i in range(total_num//volume_size):
        vid_list.append( idx_base + volume_size * i )
    return np.asarray(vid_list)

# 创建新的NIfTI文件
def create_nifti(data, affine=None, header=None):
    """创建新的NIfTI文件, 适用于Nifti1"""
    if affine is None:
        affine = np.eye(4)
    
    if header is None:
        header = nib.Nifti1Header()
    
    new_img = nib.Nifti1Image(data, affine, header)
    return new_img

# 检查NIfTI版本
def check_nifti_version(img):
    if isinstance(img, nib.Nifti1Image):
        return "NIfTI-1"
    elif isinstance(img, nib.Nifti2Image):
        return "NIfTI-2"
    else:
        return "Unknown"
    
# 检查header元数据
def check_nifti_header(img):
    header = img.header
    for k in header:
        print(f"header {k}: {header[k]}")

def update_header_info(img, updates):
    """更新头文件信息"""
    header = img.header.copy()
    for key, value in updates.items():
        header[key] = value
    return nib.Nifti1Image(img.get_fdata(), img.affine, header)

def check_and_create(path):
    if not os.path.exists(path):
        os.makedirs(path)
        return f"Created: {path}"
    elif not os.path.isdir(path):
        return f"Error: {path} exists but is not a directory"
    else:
        return f"Already exists: {path}"

def split_ori_to_new_files(fpath, ratio, args):
    """ from subject path get .nii file, save to a dict 
    input:
        fpath: str, subject_0001's path (dir)
        ratio: int, split into N files, N=num_subs/ratio
    """
    subj = os.path.basename(fpath)
    niifs = [f for f in os.listdir(fpath) if f.endswith('.nii')]

    if len(niifs) !=1:
        print("WARNING!!!   subject {} has not just 1 nii file".format(subj))
    else:
        print('Processing: ', subj, niifs[0])
        niif = niifs[0]
        nii_img = nib.load(os.path.join(fpath,niif))

        data = nii_img.get_fdata()
        header = nii_img.header
        affine = nii_img.affine

        st = 0
        step = data.shape[3]//ratio # time dim, split to two parts
        data_1 = data[:,:,:,:st+step] #TODO: wrap this with for block
        data_2 = data[:,:,:,st+step:]
        img_1 = create_nifti(data_1, affine=affine, header=header)
        img_2 = create_nifti(data_2, affine=affine, header=header)

        updates = {'dim': np.array([4,64,64,40,header['dim'][4]//ratio,1,1,1],dtype='int16')}
        update_header_info(img_1, updates)
        update_header_info(img_2, updates)
        
        # save new nii files
        save_path_1 = os.path.join(args.data_root,'CFH_expand', args.surg_time+'_01', 'Rest', subj+'_01')
        save_path_2 = os.path.join(args.data_root,'CFH_expand', args.surg_time+'_02', 'Rest', subj+'_02')
        check_and_create(save_path_1)
        check_and_create(save_path_2)
        fname_1 = os.path.join(save_path_1, niif.split('.')[0]+'.nii')
        fname_2 = os.path.join(save_path_2, niif.split('.')[0]+'.nii')
        nib.save(img_1, fname_1)
        nib.save(img_2, fname_2)

def main_for_split_ori_to_new_files():
    args = argument_parser()
    ori_bold_path = os.path.join(args.data_root, 'CFH_origin', args.surg_time, args.surg_time+'_BOLD')
    ori_t1_path = os.path.join(args.data_root, 'CFH_origin', args.surg_time, args.surg_time+'_T1')

    if args.data_phase == 'Rest':
        # process BOLD data, split 1 into 2 files
        list_subj_path = pp.get_1ring_subdirs(ori_bold_path)
        list_subj_path = sorted(list_subj_path, key=pp.natural_sort_key)

        for subj_path in list_subj_path:
            split_ori_to_new_files(subj_path, ratio=2, args=args)
        print('finished split bold files')
    elif args.data_phase == 'Struc':
        # precess T1 data, copy into folder
        tar_01_path = os.path.join(args.data_root, 'CFH_expand', args.surg_time+'_01', 'Struc')
        tar_02_path = os.path.join(args.data_root, 'CFH_expand', args.surg_time+'_02', 'Struc')
        shutil.copytree(ori_t1_path, tar_01_path)
        shutil.copytree(ori_t1_path, tar_02_path)
        print('finished copy T1 files')
    else:
        raise NotImplementedError("data phase shold be in [Rest, Struct]")

def main_2():
    args = argument_parser()

    print("----Start----")
    # script: process all data in remote PC
    
    # main_for_split_ori_to_new_files()
    args.data_root = '/mnt/c/Works/ws/shoufa2025/data/sub_0065_0066_bold'
    ori_bold_path = os.path.join(args.data_root)

    if args.data_phase == 'Rest':
        # process BOLD data, split 1 into 2 files
        list_subj_path = pp.get_1ring_subdirs(ori_bold_path)
        list_subj_path = sorted(list_subj_path, key=pp.natural_sort_key)

        for subj_path in list_subj_path:
            split_ori_to_new_files(subj_path, ratio=2, args=args)
        print('finished split bold files')
    else:
        raise NotImplementedError("data phase shold be in [Rest, Struct]")


# # 202507 process nifti file directly
def split_nifti_by_odd_even_volumes(input_nifti_path, output_dir, slices_per_volume=40):
    """
    将 NIfTI 文件按照 volume 的奇偶性分割成两个新的 NIfTI 文件
    
    参数:
        input_nifti_path: 输入 NIfTI 文件路径
        output_dir: 输出目录
        slices_per_volume: 每个 volume 包含的 slice 数量
    """
    # 创建输出目录
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # 加载 NIfTI 文件
    print(f"正在加载 {input_nifti_path}...")
    nifti_img = nib.load(input_nifti_path)
    
    # 获取数据和头信息
    data = nifti_img.get_fdata()
    header = nifti_img.header
    affine = nifti_img.affine
    
    # 打印基本信息
    print(f"图像维度: {data.shape}")  # (64,64,40,300)
    print(f"数据类型: {data.dtype}")  # float64
    
    # 计算 volume 总数
    total_volumes = data.shape[3]
    print(f"总 volume 数: {total_volumes} (每个 volume 包含 {slices_per_volume} 个 slice)")
    
    # 创建奇偶 volume 的空数组
    # 确定输出数组的形状
    odd_shape = list(data.shape)
    even_shape = list(data.shape)
    
    # 计算奇数和偶数 volume 的数量
    odd_volumes_count = total_volumes // 2 + (1 if total_volumes % 2 == 1 else 0)
    even_volumes_count = total_volumes // 2
    
    # 设置输出数组的第三维度(volume维度)
    odd_shape[3] = odd_volumes_count
    even_shape[3] = even_volumes_count
    
    # 创建输出数组
    odd_data = np.zeros(odd_shape, dtype=data.dtype)
    even_data = np.zeros(even_shape, dtype=data.dtype)
    
    # 分割数据
    odd_idx = 0
    even_idx = 0
    
    print(f"开始分割数据...")
    for vol_idx in tqdm(range(total_volumes), desc="处理 volume"):
        # 计算当前 volume 的切片范围
        start_vol = vol_idx
        end_vol = vol_idx + 1
        
        # 提取当前 volume 的所有切片
        volume_data = data[:, :, :, start_vol:end_vol]
        
        # 根据 volume 的奇偶性，将其放入相应的数组
        if (vol_idx+1) % 2 == 0:  # 奇数 volume (从0开始计数，0为第一个volume，是偶数)
            odd_start = odd_idx
            odd_end = odd_start + 1
            odd_data[:, :, :, odd_start:odd_end] = volume_data
            odd_idx += 1
        else:  # 偶数 volume
            even_start = even_idx
            even_end = even_start + 1
            even_data[:, :, :, even_start:even_end] = volume_data
            even_idx += 1
    
    # 创建并保存新的 NIfTI 文件
    print(f"创建奇数 volume 的 NIfTI 文件...")
    odd_nifti = nib.Nifti1Image(odd_data, affine, header)
    odd_output_path = os.path.join(output_dir, os.path.basename(input_nifti_path).replace('.nii', '_odd.nii'))
    nib.save(odd_nifti, odd_output_path)
    
    print(f"创建偶数 volume 的 NIfTI 文件...")
    even_nifti = nib.Nifti1Image(even_data, affine, header)
    even_output_path = os.path.join(output_dir, os.path.basename(input_nifti_path).replace('.nii', '_even.nii'))
    nib.save(even_nifti, even_output_path)
    
    # 打印结果信息
    print(f"\n处理完成!")
    print(f"奇数 volume NIfTI 文件 (共 {odd_volumes_count} 个 volumes, {odd_idx * slices_per_volume} 个 slices):")
    print(f"  保存路径: {odd_output_path}")
    print(f"  数据形状: {odd_data.shape}")
    
    print(f"\n偶数 volume NIfTI 文件 (共 {even_volumes_count} 个 volumes, {even_idx * slices_per_volume} 个 slices):")
    print(f"  保存路径: {even_output_path}")
    print(f"  数据形状: {even_data.shape}")


if __name__ == '__main__':
    # 20250707 process nifti file directly
    # 将 NIfTI 文件按照 volume 的奇偶性分割
    def extract_subject_number(path):
        """
        从路径中提取sub_XXXX格式的数字部分
        例如从'/mnt/f/CFH_Original_Data/sub_0053_1040368_wu_yong_liang/...'提取出53
        
        Args:
            path: 包含子目录的路径
        
        Returns:
            int: 提取的数字，如果没有找到则返回float('inf')作为排序的默认值
        """
        # 使用正则表达式查找sub_后跟随的数字部分
        match = re.search(r'sub_(\d+)_', path)
        if match:
            # 将提取的数字字符串转换为整数
            return int(match.group(1))
        else:
            # 如果没有找到匹配，返回无穷大以便排在最后
            return float('inf')

    def find_qualifying_directories(root_dir):
        """
        遍历指定目录，找出符合以下条件的子目录：
        1. 目录名称以'AxBOLD'结尾
        2. 为最末级子目录（不包含子目录）
        
        Args:
            root_dir: 根目录路径
        
        Returns:
            list: 符合条件的目录绝对路径列表
        """
        qualifying_dirs = []
        
        # 计数器用于显示进度
        total_dirs = 0
        processed_dirs = 0
        
        # 首先统计总目录数以便显示进度
        print("计算总目录数...")
        for _, dirnames, _ in os.walk(root_dir):
            total_dirs += len(dirnames)
        
        print(f"开始处理，总共 {total_dirs} 个目录...")
        
        # 遍历所有目录
        for dirpath, dirnames, filenames in os.walk(root_dir):
            processed_dirs += 1

            # 每处理100个目录显示一次进度
            if processed_dirs % 100 == 0:
                progress = (processed_dirs / total_dirs) * 100
                print(f"进度: {progress:.2f}% ({processed_dirs}/{total_dirs})")
            
            # 检查1: 目录名是否以'AxBOLD'结尾
            if not os.path.basename(dirpath).endswith('AxBOLD'):
                continue
            
            # 检查2: 是否为最末级子目录（不包含其他子目录）
            if dirnames:  # 如果dirnames不为空，说明有子目录
                continue

            qualifying_dirs.append(os.path.abspath(dirpath))

        return sorted(qualifying_dirs, key=extract_subject_number)
    
    src_dir_list = find_qualifying_directories(root_dir='/mnt/c/Works/ws/shoufa2025/data/CFH_original_2507')
    tar_dir_list = [path.replace('original_2507', 'processed_2507') for path in src_dir_list]

    for i,src_dir in enumerate(src_dir_list):
        tar_dir = tar_dir_list[i]
        input_nii_file = glob.glob(os.path.join(src_dir, "*.nii"))[0]
        split_nifti_by_odd_even_volumes(input_nifti_path=input_nii_file, output_dir=tar_dir, slices_per_volume=40)
