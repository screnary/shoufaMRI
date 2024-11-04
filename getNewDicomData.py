""" Expand origin dicom data through random volume expantion,
    Output new dicom files
"""
import os
import numpy as np
from collections import defaultdict
import cv2
import matplotlib.pyplot as plt
import pydicom
from pydicom.dataset import Dataset, FileDataset
from tqdm import tqdm
import preprocess_data as pp
from datetime import datetime
import shutil

import nibabel as nib
import pdb


proj_root = os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))


def argument_parser():
    import argparse
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
    niifs = os.listdir(fpath)
    assert len(niifs) ==1 and niifs[0].endswith('.nii'), "subject {} has multiple files, or file is not .nii".format(subj)
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

if __name__ == '__main__':
    args = argument_parser()

    print("----Start----")
    # script: process all data in remote PC
    
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
        shtil.copytree(ori_t1_path, tar_01_path)
        shtil.copytree(ori_t1_path, tar_02_path)
        print('finished copy T1 files')
    else:
        raise NotImplementedError("data phase shold be in [Rest, Struct]")
