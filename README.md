# shoufaMRI

## in getNewDicomData.py, process dicom image slices [2024.09 bkup]
```python
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
    
    # S1.3 查看slice数据的关键meta data并保存备用, 构建新的Dicom数据时使用
        
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
```

CFH_origin 分割为两半，在mac上少量数据进行测试
```python
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
    
    def split_ori_to_new_files(fpath, ratio=2):
        """ from subject path get .nii file, save to a dict """
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
        data_1 = data[:,:,:,:st+step]
        data_2 = data[:,:,:,st+step:]
        img_1 = create_nifti(data_1, affine=affine, header=header)
        img_2 = create_nifti(data_2, affine=affine, header=header)

        updates = {'dim': np.array([4,64,64,40,header['dim'][4]//ratio,1,1,1],dtype='int16')}
        update_header_info(img_1, updates)
        update_header_info(img_2, updates)
        
        # save new nii files
        save_path_1 = os.path.join(root_dir, 'Data', 'CFH_expand', 'Post_Surgery', 'Post_Surgery_BOLD', subj+'_01')
        save_path_2 = os.path.join(root_dir, 'Data', 'CFH_expand', 'Post_Surgery', 'Post_Surgery_BOLD', subj+'_02')
        check_and_create(save_path_1)
        check_and_create(save_path_2)
        fname_1 = os.path.join(save_path_1, niif.split('.')[0]+'_01'+'.nii')
        fname_2 = os.path.join(save_path_2, niif.split('.')[0]+'_02'+'.nii')
        nib.save(img_1, fname_1)
        nib.save(img_2, fname_2)

    for subj_path in list_subj_path:
        split_ori_to_new_files(subj_path)
        print('*********** finished ************')
```

在家里PC上，全量数据处理，CFH_origin to CFH_expand
要求是：1）_001和_002分开文件夹放，分别建立一个以_001和_002为内容的文件夹；2）在这两个文件夹里都放入T1内容(直接复制进去即可) 3）nii数据命名无要求，故沿用原名
CFH_expand/Post_Surgery_001/Rest/sub_0001_01/*.nii
CFH_expand/Post_Surgery_001/Struc/sub_0001_01/*.nii
CFH_expand/Post_Surgery_002/Rest/sub_0001_02/*.nii
CFH_expand/Post_Surgery_002/Struc/sub_0001_02/*.nii
```shell
# Post_Surgery, Rest
WARNING!!! sub_0019, sub_0021 has no file

# Pre_Surgery, Rest
WARNING!!! sub_0004, sub_0027, sub_0028, sub_0033~0037 has no file
```

## POE nifti
NIfTI (Neuroimaging Informatics Technology Initiative) 数据格式是神经影像领域常用的文件格式。以下是其主要特征和结构：

1. 基本结构：
```python
import nibabel as nib
import numpy as np

# 加载NIfTI文件
img = nib.load('example.nii')

# NIfTI文件的三个主要组成部分
data = img.get_fdata()  # 图像数据
header = img.header    # 头文件信息
affine = img.affine   # 仿射矩阵

# 头文件主要信息
print(f"维度: {header['dim']}") 
print(f"体素大小: {header['pixdim']}")
print(f"数据类型: {header.get_data_dtype()}")
print(f"图像尺寸: {data.shape}")
```

2. 主要头文件字段：
```python
# 常见的头文件信息字段
important_fields = {
    'dim': '数据维度和每个维度的大小',
    'pixdim': '体素在每个维度的物理尺寸(mm)',
    'datatype': '数据类型编码',
    'qform_code': '空间变换类型代码',
    'sform_code': '空间变换类型代码',
    'slice_start': '体积扫描的起始切片索引',
    'slice_end': '体积扫描的结束切片索引',
    'slice_duration': '获取每个切片的时间',
    'toffset': '时间单位偏移',
    'descrip': '数据描述信息'
}

# 查看特定字段
for field in important_fields:
    if field in header:
        print(f"{field}: {header[field]}")
```

3. 仿射变换矩阵：
```python
# 仿射矩阵定义了体素坐标到物理空间的映射
print("仿射矩阵:")
print(affine)

# 坐标变换示例
def voxel_to_world(voxel_coords, affine):
    """体素坐标转世界坐标"""
    coords = np.array(voxel_coords + [1])
    world_coords = np.dot(affine, coords)
    return world_coords[:3]

def world_to_voxel(world_coords, affine):
    """世界坐标转体素坐标"""
    inv_affine = np.linalg.inv(affine)
    coords = np.array(world_coords + [1])
    voxel_coords = np.dot(inv_affine, coords)
    return voxel_coords[:3]
```

4. 数据存储格式：
```python
# 数据类型
data_types = {
    2: 'uint8',
    4: 'int16',
    8: 'int32',
    16: 'float32',
    32: 'complex64',
    64: 'float64',
    256: 'int8',
    512: 'uint16',
    768: 'uint32'
}

# 创建新的NIfTI文件
def create_nifti(data, affine=None, header=None):
    """创建新的NIfTI文件"""
    if affine is None:
        affine = np.eye(4)
    
    if header is None:
        header = nib.Nifti1Header()
    
    new_img = nib.Nifti1Image(data, affine, header)
    return new_img
```

5. 文件格式版本：
```python
# NIfTI-1 格式示例
nii1_img = nib.Nifti1Image(data, affine)

# NIfTI-2 格式示例
nii2_img = nib.Nifti2Image(data, affine)

# 检查NIfTI版本
def check_nifti_version(img):
    if isinstance(img, nib.Nifti1Image):
        return "NIfTI-1"
    elif isinstance(img, nib.Nifti2Image):
        return "NIfTI-2"
    else:
        return "Unknown"
```

6. 文件操作工具：
```python
def save_nifti(img, filename):
    """保存NIfTI文件"""
    nib.save(img, filename)

def copy_header(source_img, target_data):
    """复制头文件信息"""
    new_img = nib.Nifti1Image(target_data, 
                             source_img.affine, 
                             source_img.header)
    return new_img

def validate_nifti(filename):
    """验证NIfTI文件"""
    try:
        img = nib.load(filename)
        # 基本检查
        assert img.header.get_data_dtype() != object
        assert len(img.header['dim']) == 8
        assert img.get_fdata().size > 0
        return True
    except Exception as e:
        print(f"验证失败: {str(e)}")
        return False
```

7. 常见操作：
```python
def get_orientation(img):
    """获取图像方向信息"""
    return nib.aff2axcodes(img.affine)

def reorient_nifti(img, target_orientation='RAS'):
    """重新定向图像"""
    return nib.as_closest_canonical(img)

def resample_nifti(img, new_voxel_size):
    """重采样到新的体素大小"""
    from dipy.align.reslice import reslice
    
    data = img.get_fdata()
    zooms = img.header.get_zooms()
    
    new_data, new_affine = reslice(data, 
                                  img.affine, 
                                  zooms, 
                                  new_voxel_size)
    
    return nib.Nifti1Image(new_data, new_affine)
```

8. 元数据处理：
```python
def update_header_info(img, updates):
    """更新头文件信息"""
    header = img.header.copy()
    for key, value in updates.items():
        header[key] = value
    return nib.Nifti1Image(img.get_fdata(), img.affine, header)

def extract_metadata(img):
    """提取重要元数据"""
    metadata = {
        'dimensions': img.header['dim'][1:],
        'voxel_sizes': img.header['pixdim'][1:],
        'data_type': img.header.get_data_dtype(),
        'orientation': nib.aff2axcodes(img.affine),
        'units': img.header.get_xyzt_units(),
        'description': img.header['descrip'].tostring().decode('utf-8').strip('\x00')
    }
    return metadata
```

NIfTI格式的主要特点：

1. 支持多维数据（通常是3D或4D）
2. 包含详细的头文件信息
3. 支持仿射变换定义空间关系
4. 广泛用于医学影像和神经影像研究
5. 可以存储为单个.nii文件或压缩的.nii.gz文件
6. 支持多种数据类型和存储格式

使用注意事项：

1. 确保正确处理坐标系统和方向信息
2. 注意数据类型的精度和范围
3. 处理大文件时需要考虑内存使用
4. 保存时考虑是否需要压缩
5. 注意保持元数据的完整性

如果你有特定的NIfTI数据处理需求，我可以提供更具体的代码示例。
