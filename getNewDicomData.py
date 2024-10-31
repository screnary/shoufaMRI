""" Expand origin dicom data through random volume expantion,
    Output new dicom files
"""
import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
import pydicom
from pydicom.dataset import Dataset, FileDataset
from tqdm import tqdm
import preprocess_data as pp
from datetime import datetime
import nibabel as nib
import pdb


proj_root = os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))


def argument_parser():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_root', type=str,
                        default="/mnt/e/data/liuyang/2024shoufa",
                        help='data root for this project')
    
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
    """创建新的NIfTI文件"""
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


if __name__ == '__main__':

    print("----Start----")
    # args = argument_parser()
    # test_fpath = os.path.join(args.data_root, '1000814任俊杰/DICOM/PA0/ST0/SE5')
    # imlist = sorted(os.listdir(test_fpath),key=pp.natural_sort_key)
    
    # fn = os.path.join(test_fpath, imlist[0])
    # parseDicomFile(fn)
    
    # Step1. 处理样例数据，按文件序列组织为 Volume instance 并查看数据正确性，中间数据存储
    # S1.1 将volume与文件的序列关系存储为二维数组
    root_dir = os.path.dirname(os.path.abspath(__file__))
    cfh_origin_path = os.path.join(root_dir, 'Data', 'CFH_origin')
    BOLD_path = os.path.join(cfh_origin_path, 'Post_Surgery', 'Post_Surgery_BOLD')
    list_subj_path = pp.get_1ring_subdirs(BOLD_path)
    list_subj_path = sorted(list_subj_path, key=pp.natural_sort_key)
    list_subj_name = [os.path.basename(path) for path in list_subj_path]
    list_subj_name = sorted(list_subj_name, key=pp.natural_sort_key)
    
    def get_subj_nii_pairs(fpath):
        """ from subject path get .nii file, save to a dict """
        subj = os.path.basename(fpath)
        niif = os.listdir(fpath)
        assert len(niif) ==1 and niif[0].endswith('.nii'), "subject {} has multiple files, or file is not .nii".format(subj)
        print(subj, niif)
        nii_img = nib.load(os.path.join(fpath,niif[0]))
        # check nifti version
        check_nifti_version(nii_img)
        # read nii data, header, affine matrix
        data = nii_img.get_fdata()
        header = nii_img.header
        affine = nii_img.affine
        print("data shape: {}".format(data.shape))
        print("data dtype: {}".format(data.dtype))
        print("header voxel size: {}".format(header.get_zooms()))
        pdb.set_trace()
        

    get_subj_nii_pairs(list_subj_path[2])
