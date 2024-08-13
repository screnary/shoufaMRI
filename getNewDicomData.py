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

update_dicom(input_file, output_file, updates)
    

if __name__ == '__main__':
    print("----Start----")
    args = argument_parser()
    test_fpath = os.path.join(args.data_root, '1000814任俊杰/DICOM/PA0/ST0/SE5')
    imlist = sorted(os.listdir(test_fpath),key=pp.natural_sort_key)
    
    fn = os.path.join(test_fpath, imlist[0])
    parseDicomFile(fn)
