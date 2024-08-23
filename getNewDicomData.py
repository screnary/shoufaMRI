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
    config_fpath = os.path.join(root_dir, 'Data', 'volume_idx_mapping.npy')
    if os.path.isfile(config_fpath):
        vid_list = np.load(config_fpath)
        print("config file loaded")
    else:
        print("config file not found, create now...")
        vid_list = getVidIdx(volume_size=40, total_num=12000) # get index of stack of volumes [volume1[s1,s2,...],volume2,...]
        np.save(config_fpath, vid_list)
    
    # S1.2 将Dicom数据处理为 volume 数据结构; ? 写完该部分函数，将它移到 preprocess_data 中
    imfolder = os.path.join(root_dir, 'Data', 'SE4')
    def cvDicom2Volume(imfolder=imfolder, vid_list=vid_list):
        # read DICOM files--BOLD files, and process them to Volume data structure
        # setup dirs, can be read from config inputs
        # print("******************\n\tProcessing {} {}\n******************".format(args.group, args.time_stamp))
        
        # get all IM files, to a fname list
        im_list = sorted(os.listdir(imfolder),key=pp.natural_sort_key)
        slice_list = [im_list[im_id] for im_id in vid_list[0]]

        volume = pp.getVolume(slice_list, imfolder, save_path=os.path.join(root_dir, 'Data', 'SE4_test'), v_id=0)
        volume.save2img()
        print('img saved')

    cvDicom2Volume()
        
        # instance_path_list = sorted(get_1ring_subdirs(group_root))
        # for (i, instance_path) in enumerate(instance_path_list):
        #     instance_name = instance_path.split('/')[-1]
        #     dicom_phase_list = sorted(get_1ring_subdirs(instance_path))
        #     dicom_slice_path = dicom_phase_list[1]
        #     dicom_slice_suffix = dicom_slice_path.split('__')[-1]
        #     dicom_bold_path = dicom_phase_list[-1]
        #     dicom_bold_suffix = dicom_bold_path.split('__')[-1]
        #     slice_file = [f for f in sorted(os.listdir(dicom_slice_path))
        #                 if not f.startswith(".")]
        #     bold_file = [f for f in sorted(os.listdir(dicom_bold_path))
        #                 if f.startswith("MRI")]
        #     if not (len(slice_file) == args.slice_num and len(bold_file) == args.bold_num):
        #         print("dicom file num NOT VALID", len(slice_file), len(bold_file))
        #         pdb.set_trace()

        #     print("Processing instance {}, [{}/{}]...".format(instance_path.split('/')[-1], i+1, len(instance_path_list)))

        #     print("Processing Slices High Resolution...")
        #     save_path = os.path.join(args.save_path, instance_name, 'pose_fix-' + dicom_slice_suffix)
        #     fns = slice_file
        #     volume = getVolume(fns, dicom_slice_path, save_path, v_id=0)
        #     volume.save2img()

        #     if not os.path.exists(save_path):
        #         os.makedirs(save_path)
        #     with open(os.path.join(save_path, 'volume.npz'), 'wb') as f:
        #         vdata = (volume.vdata * 255).astype('uint8')
        #         np.savez(f, vdata)

            # print("Processing volumes...")
            # s_num = args.slice_num
            # v_num = args.bold_num // args.slice_num
            # save_path = os.path.join(args.save_path, instance_name, 'bold-' + dicom_bold_suffix)
            # for j in tqdm(range(v_num)):
            #     st, ed = j * s_num, (j + 1) * s_num
            #     fns = bold_file[st:ed]
            #     try:
            #         volume = getVolume(fns, dicom_bold_path, save_path, v_id=j+1)
            #         volume.save2img()
            #         if not os.path.exists(save_path):
            #             os.makedirs(save_path)
            #         with open(os.path.join(save_path, 'volume_{:05d}.npz'.format(j+1)), 'wb') as f:
            #             vdata = (volume.vdata * 255).astype('uint8')
            #             np.savez(f, vdata)
            #     except Exception as e:
            #         print("Error: ", e)
            #         print("An Error Occurred while processing volume_{}".format(j+1))
    # Step2. 将 volume structure 解析存储为 Dicom IM 数据
    
    
    pdb.set_trace()
            
    
