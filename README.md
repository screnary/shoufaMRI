# shoufaMRI Env Setting

## Install Project
```shell
git clone git@github.com:screnary/shoufaMRI.git
```

## Project directories
``` shell
tree -L 2



├── README.md
├── __init__.py
├── __pycache__
│   ├── __init__.cpython-310.pyc
│   ├── __init__.cpython-312.pyc
│   └── preprocess_data.cpython-312.pyc
├── config
│   ├── config_20250730.yaml
│   └── config_default.yaml
├── core
│   ├── __init__.py
│   ├── __pycache__
│   ├── exp_frame_noise.py
│   └── volume_noise_operations.py
├── data
│   ├── __init__.py
│   ├── dicom_handler.py
│   ├── extract_matrix.py
│   ├── file_merger.py
│   ├── file_organizer.py
│   └── preprocess_data.py
├── requirements.txt
├── run_noise_experiments.py
├── run_noise_processing_test.py
├── run_process_command.py
├── run_split_original.py
├── run_visualize_3d.py
└── utils
    ├── __init__.py
    ├── __pycache__
    ├── config_loader.py
    ├── directory_utils.py
    └── visualize.py
```

## Requirements
```shell
pip freeze | grep -E "(numpy|matplotlib|scipy|opencv|pillow|pydicom|nibabel|scikit)" > requirements.txt

matplotlib==3.10.1
matplotlib-inline==0.1.7
nibabel==5.3.2
numpy==2.2.5
opencv-python==4.11.0.86
pillow==11.2.1
pydicom==3.0.1
scikit-image==0.25.2
scipy==1.15.3
openpyxl
pandas
```


## Visualization in Windows [miniconda]
### 1. 下载并安装Anaconda
访问: https://www.anaconda.com/products/distribution
下载Windows版本并安装

配置环境变量：
```
1. Win + R，输入 sysdm.cpl
2. 点击"高级"选项卡
3. 点击"环境变量"
4. 配置Path变量，添加
C:\Users\你的用户名\miniconda3
C:\Users\你的用户名\miniconda3\Scripts
C:\Users\你的用户名\miniconda3\Library\bin
```

验证配置：
```shell
#REM 检查conda是否可用
conda --version

#REM 检查conda信息
conda info

#REM 检查环境变量
echo $env:Path
```


### 2. 打开Anaconda Prompt（以管理员身份）

### 3. 创建专用环境
``` shell
conda create -n brain_viz python=3.10 -y
conda activate brain_viz
```
### 4. 安装核心包
``` shell
conda install -c conda-forge pyvista vtk numpy matplotlib nibabel jupyter jupyterlab -y
```
### 5. 安装额外的可视化包
``` shell
conda install -c conda-forge panel ipywidgets trame -y
```
### 6. 测试安装
``` shell
python -c "
import pyvista as pv
import numpy as np
```
### 创建交互式可视化
``` shell
sphere = pv.Sphere()
sphere.plot(color='lightblue', show_edges=True)
print('✓ Windows PyVista 工作正常!')
```

## 修改记录 (Changelog)

### [2026-01] BuildingML批量矩阵提取

**功能**: 为BuildingML进行批量matrix提取

- **数据目录**: `C:\Works\ws\shoufa2025\data\202512_BuidingML`
- **输出目录**: `C:\Works\ws\shoufa2025\data\matrix\arrnaged_Whole_Brain`
- **功能描述**: 从全脑文件夹中提取每个区域网络的子矩阵，生成对应文件夹（例如：`02_MoCA_unchanged/zsub_0004.txt`）

**使用方法**:
```shell
cd /mnt/c/Works/ws/shoufa2025/code/shoufaMRI/data
python extract_matrix.py
```

---

### [2025-12] 添加噪声后数据的子网络提取

**修复**: DAN、DMN错误提取修正
- 运行 `extract_matrix.py` 进行矩阵提取
- 运行 `copy_to_category_folder.py` 进行分类拷贝
- **数据存储**: `C:\Works\ws\shoufa2025\data\matrix\原始数据_子网提取\matrix_DAN_DMN_修正.rar`

**功能改进**: 添加噪声后数据的子网络提取
1. noised_matrix_data中，3类数据增强子文件夹，定位到最内层目录，可能还要处理文件名的模式变化“zsub_0001_01.txt”
1） sub标识添加响应处理
2） 目录遍历，以及输出目录【可以不做这块，把中间目录略掉】


---

### [2025-11] 新增功能模块
## 添加 data/extract_matrix.py
1. 读取并解析 excel 的特定 sheet，提取其中标黄/高亮位置的矩阵坐标
2. 根据矩阵坐标，从zsub_0xxx.txt中提取对应位置的子矩阵，存储为新矩阵
3. 将子矩阵按照 subnect 的 CAM 分类存到各自的目录

## 添加 data/copy_to_category_folder.py
Step 1. - 提取CAM分类：

从Excel的"剔除无MRI"sheet中读取数据
按"CAM评分术后"列分类
提取每个类别对应的"MRI排序"列表
Step 2. - 拷贝文件：

根据CAM分类结果
在源目录中查找匹配的zsub_****_submatrix.txt文件
按类别拷贝到对应的文件夹
类别名称映射：

默认映射：正常→normal, 谵妄→delirium, 不全对→incomplete, 待数据→pending_data
可以自定义映射
文件匹配逻辑：

支持zsub_0001_submatrix.txt和sub_0001_submatrix.txt格式
大小写不敏感匹配
自动处理前缀z

category_name_mapping = {
        '正常': 'normal',
        '谵妄': 'delirium',
        '不全对': 'incomplete',
        '待数据': 'pending_data',
        '患者出院就算正常吧哈哈哈哈': 'normal'
    }

---

### [2025-10] 原始数据处理实验

**实验**: Process original_202510
1. 处理目录，
- [x] (1) 从中取出：E:\data\liuyang\original_202510\for_original\Rest_pre\sub_0001，中的bcNGS前缀的.nii文件，存到一个新目录“for_original_bcNGS”
```shell
python directory_utils.py /mnt/e/data/liuyang/original_202510/for_original/Rest_pre /mnt/e/data/liuyang/original_202510/for_original_bcNGS/Rest_pre --pattern "bcNGS*.nii"

python directory_utils.py /mnt/e/data/liuyang/original_202510/for_original/Rest_post /mnt/e/data/liuyang/original_202510/for_original_bcNGS/Rest_post --pattern "bcNGS*.nii"
```
- [x] (2) 从中取出：E:\data\liuyang\original_202510\for_original\Rest_pre\sub_0001，中的无前缀的20240416_072804LIUYANGFMRIs002a1001.nii文件,存到一个新目录“for_original_noprocess”
```shell
python directory_utils.py /mnt/e/data/liuyang/original_202510/for_original/Rest_pre /mnt/e/data/liuyang/original_202510/for_original_noprocess/Rest_pre --pattern "[0-9]*.nii"

python directory_utils.py /mnt/e/data/liuyang/original_202510/for_original/Rest_post /mnt/e/data/liuyang/original_202510/for_original_noprocess/Rest_post --pattern "[0-9]*.nii"
```

2. 按照 Experiment 202509 [Band pass noise] 配置进行带通滤波加噪
- [x] (1) 更新 protected mask 的 anchor MNI coordinates
- [x] (2) 运行 run_noise_experiments.py，存到目录“for_original_bcNGS_bandpass_noised”
``` shell
python run_noise_experiments.py -n "[20251104] r=12 and std=[25,100]" -c "./config/config_20251104_pre.yaml"

python run_noise_experiments.py -n "[20251104] r=12 and std=[25,100]" -c "./config/config_20251104_post.yaml"
```


3. 将原始.nii间隔采样、对半采样
- [x]  (1) 运行 dicom_handler.py 中的 split_nifti_by_odd_even_volumes(), 存到目录“for_original_noprocess_interval”
- [x] (2) 运行 dicom_handler.py 中的 split_ori_to_two_files(), 存到目录“for_original_noprocess_half”
``` shell
cd data
python dicom_handler.py
# using main_process_nifti_202511() function
```


---

### [2025-09] 带通噪声实验

**实验**: Band pass noise
1. 在 core/volume_noise_operations.py 中，进行带通噪声实验，验证添加噪声的频率是否符合要求。
2. 修改 其中 add_noise_avoid_coordinates() 函数，让其可选进行带通噪声添加。
3. 修改 exp_frame_noise.py 和 config，将带通噪声参数加进去，进而进行批量处理。[不需，直接在add_noise_avoid_coordinates使用了默认参数，默认使用带通滤波处理noise]
4. 直接批量运行之前config即可（改data experiment名称）
``` shell
python run_noise_experiments.py -n "[20250911] r=12 and std=[25,100]" -c "./config/config_20250911.yaml"
```

``` shell
设置包名: shoufaMRI
加载配置文件: ./config/config_20250911.yaml
使用坐标组: whole
开始实验...
2025-09-11 21:07:43,324 - [20250911] r=12 and std=[25,100], pre - INFO - 实验配置已保存: /mnt/e/data/liuyang/original_202507/experiments/[20250911] r=12 and std=[25,100], pre/config.json
2025-09-11 21:07:43,324 - [20250911] r=12 and std=[25,100], pre - INFO - 开始实验: [20250911] r=12 and std=[25,100], pre
2025-09-11 21:07:43,781 - [20250911] r=12 and std=[25,100], pre - INFO - 共处理 64/64 个NIfTI文件
2025-09-11 21:07:43,782 - [20250911] r=12 and std=[25,100], pre - INFO - 生成 2 个参数组合
2025-09-11 21:07:43,782 - [20250911] r=12 and std=[25,100], pre - INFO - 总任务数: 128

2025-09-11 21:08:01,716 - [20250911] r=12 and std=[25,100], pre - ERROR - 任务异常: /mnt/e/data/liuyang/original_202507/04_original_data_no_editing_processed/pre_surgery/RestTARWSDCF/sub_0083/sub_0083F.nii, A process in the process pool was terminated abruptly while the future was running or pending.
2025-09-11 21:08:01,716 - [20250911] r=12 and std=[25,100], pre - ERROR - 任务异常: /mnt/e/data/liuyang/original_202507/04_original_data_no_editing_processed/pre_surgery/RestTARWSDCF/sub_0083/sub_0083F.nii, A process in the process pool was terminated abruptly while the future was running or pending.
2025-09-11 21:08:01,717 - [20250911] r=12 and std=[25,100], pre - ERROR - 任务异常: /mnt/e/data/liuyang/original_202507/04_original_data_no_editing_processed/pre_surgery/RestTARWSDCF/sub_0084/sub_0084F.nii, A process in the process pool was terminated abruptly while the future was running or pending.
2025-09-11 21:08:01,717 - [20250911] r=12 and std=[25,100], pre - ERROR - 任务异常: /mnt/e/data/liuyang/original_202507/04_original_data_no_editing_processed/pre_surgery/RestTARWSDCF/sub_0084/sub_0084F.nii, A process in the process pool was terminated abruptly while the future was running or pending.
2025-09-11 21:08:01,718 - [20250911] r=12 and std=[25,100], pre - ERROR - 任务异常: /mnt/e/data/liuyang/original_202507/04_original_data_no_editing_processed/pre_surgery/RestTARWSDCF/sub_0085/sub_0085F.nii, A process in the process pool was terminated abruptly while the future was running or pending.
2025-09-11 21:08:01,718 - [20250911] r=12 and std=[25,100], pre - ERROR - 任务异常: /mnt/e/data/liuyang/original_202507/04_original_data_no_editing_processed/pre_surgery/RestTARWSDCF/sub_0085/sub_0085F.nii, A process in the process pool was terminated abruptly while the future was running or pending.
2025-09-11 21:08:01,718 - [20250911] r=12 and std=[25,100], pre - ERROR - 任务异常: /mnt/e/data/liuyang/original_202507/04_original_data_no_editing_processed/pre_surgery/RestTARWSDCF/sub_0086/sub_0086F.nii, A process in the process pool was terminated abruptly while the future was running or pending.
2025-09-11 21:08:01,719 - [20250911] r=12 and std=[25,100], pre - ERROR - 任务异常: /mnt/e/data/liuyang/original_202507/04_original_data_no_editing_processed/pre_surgery/RestTARWSDCF/sub_0086/sub_0086F.nii, A process in the process pool was terminated abruptly while the future was running or pending.
2025-09-11 21:08:01,719 - [20250911] r=12 and std=[25,100], pre - ERROR - 任务异常: /mnt/e/data/liuyang/original_202507/04_original_data_no_editing_processed/pre_surgery/RestTARWSDCF/sub_0114/sub_0114F.nii, A process in the process pool was terminated abruptly while the future was running or pending.
2025-09-11 21:08:01,719 - [20250911] r=12 and std=[25,100], pre - ERROR - 任务异常: /mnt/e/data/liuyang/original_202507/04_original_data_no_editing_processed/pre_surgery/RestTARWSDCF/sub_0114/sub_0114F.nii, A process in the process pool was terminated abruptly while the future was running or pending.
2025-09-11 21:08:01,720 - [20250911] r=12 and std=[25,100], pre - ERROR - 任务异常: /mnt/e/data/liuyang/original_202507/04_original_data_no_editing_processed/pre_surgery/RestTARWSDCF/sub_0115/sub_0115F.nii, A process in the process pool was terminated abruptly while the future was running or pending.
2025-09-11 21:08:01,720 - [20250911] r=12 and std=[25,100], pre - ERROR - 任务异常: /mnt/e/data/liuyang/original_202507/04_original_data_no_editing_processed/pre_surgery/RestTARWSDCF/sub_0115/sub_0115F.nii, A process in the process pool was terminated abruptly while the future was running or pending.
2025-09-11 21:08:01,721 - [20250911] r=12 and std=[25,100], pre - ERROR - 任务异常: /mnt/e/data/liuyang/original_202507/04_original_data_no_editing_processed/pre_surgery/RestTARWSDCF/sub_0116/sub_0116F.nii, A process in the process pool was terminated abruptly while the future was running or pending.
2025-09-11 21:08:01,721 - [20250911] r=12 and std=[25,100], pre - ERROR - 任务异常: /mnt/e/data/liuyang/original_202507/04_original_data_no_editing_processed/pre_surgery/RestTARWSDCF/sub_0116/sub_0116F.nii, A process in the process pool was terminated abruptly while the future was running or pending.
2025-09-11 21:08:01,722 - [20250911] r=12 and std=[25,100], pre - ERROR - 任务异常: /mnt/e/data/liuyang/original_202507/04_original_data_no_editing_processed/pre_surgery/RestTARWSDCF/sub_0117/sub_0117F.nii, A process in the process pool was terminated abruptly while the future was running or pending.
2025-09-11 21:08:01,723 - [20250911] r=12 and std=[25,100], pre - ERROR - 任务异常: /mnt/e/data/liuyang/original_202507/04_original_data_no_editing_processed/pre_surgery/RestTARWSDCF/sub_0117/sub_0117F.nii, A process in the process pool was terminated abruptly while the future was running or pending.
2025-09-11 21:08:01,723 - [20250911] r=12 and std=[25,100], pre - ERROR - 任务异常: /mnt/e/data/liuyang/original_202507/04_original_data_no_editing_processed/pre_surgery/RestTARWSDCF/sub_0118/sub_0118F.nii, A process in the process pool was terminated abruptly while the future was running or pending.
2025-09-11 21:08:01,723 - [20250911] r=12 and std=[25,100], pre - ERROR - 任务异常: /mnt/e/data/liuyang/original_202507/04_original_data_no_editing_processed/pre_surgery/RestTARWSDCF/sub_0118/sub_0118F.nii, A process in the process pool was terminated abruptly while the future was running or pending.
2025-09-11 21:08:02,108 - [20250911] r=12 and std=[25,100], pre - INFO - 结果已保存: /mnt/e/data/liuyang/original_202507/experiments/[20250911] r=12 and std=[25,100], pre/results.csv, /mnt/e/data/liuyang/original_202507/experiments/[20250911] r=12 and std=[25,100], pre/results.json
实验失败: division by zero

# 原因，内存不够，导致pool内进程终止报错。改了worker=2即可
```

---

### [2025-07] 噪声添加实验

**实验**: Noise addition

## fMRI数据噪声添加与SNR控制方法概要

### 一、基于SNR的噪声强度控制方法

**核心原理**：通过目标信噪比反推所需噪声强度，确保统计分析的有效性。

**步骤流程**：
1. **信号功率计算**：在脑区掩膜内计算原始fMRI信号的功率（方差）
2. **目标SNR设定**：根据分析需求设置目标信噪比（一般15-25dB适合FDR分析）
3. **噪声功率推算**：利用SNR公式计算所需噪声功率和标准差
4. **噪声生成**：按计算的标准差生成高斯噪声
5. **选择性添加**：仅在脑区内添加噪声，保持背景区域不变
6. **验证检查**：计算实际SNR确认是否达到目标值

### 二、频域噪声控制策略

**核心原理**：控制噪声的频率成分，避免干扰BOLD信号的关键频率范围。

**关键频率划分**：
- **低频漂移区**：0-0.01Hz（需要避免，会影响基线）
- **BOLD信号区**：0.01-0.08Hz（需要保护，这是任务相关信号的主要频段）
- **高频噪声区**：0.08Hz以上（可以适量添加，主要是热噪声）

**实施步骤**：
1. **频率设计**：根据TR计算采样频率和奈奎斯特频率 (TR=2.0s,重复率，即fMRI采集的周期)
2. **滤波器构建**：设计带通或带阻滤波器限定噪声频段
3. **白噪声生成**：先生成宽频带的白噪声
4. **频域滤波**：对每个体素的时间序列应用频率滤波
5. **功率调整**：将滤波后的噪声调整到目标功率水平

### 三、fMRI真实噪声模拟方法

**多成分噪声策略**：模拟实际fMRI扫描中的各种噪声源。

**主要噪声成分**：
1. **热噪声**：高频均匀分布，随空间位置变化（模拟线圈敏感性）
2. **生理噪声**：包括心跳（~1Hz）和呼吸（~0.3Hz）的周期性信号
3. **扫描器漂移**：极低频成分（0.001-0.01Hz），模拟硬件不稳定性
4. **运动相关噪声**：空间相关的低到中频噪声（0.01-0.1Hz）

**空间特性考虑**：
- **距离效应**：远离中心的体素噪声更强
- **区域差异**：不同脑区对生理噪声的敏感性不同
- **空间相关性**：运动噪声在相邻体素间具有相关性

### 四、FDR兼容的噪声控制

**统计分析导向**：确保添加的噪声不会破坏FDR等多重比较校正方法的有效性。

**效应量控制**：
1. **目标效应量设定**：通常设置Cohen's d ≥ 0.5以确保检测能力
2. **噪声水平推算**：根据信号强度和目标效应量计算合适的噪声标准差
3. **统计功效评估**：预估在给定噪声水平下的统计检验功效

**验证检查项目**：
1. **正态性验证**：确认噪声符合正态分布假设
2. **频率特性检查**：验证噪声频谱不会干扰任务信号
3. **SNR合理性**：确保信噪比在统计分析的可接受范围内

### 五、推荐的实施策略

**保守策略**（推荐用于FDR分析）：
- 仅添加适量热噪声
- 目标SNR设置在20dB以上
- 严格限制低频成分
- 频繁验证统计假设

**适中策略**（一般研究使用）：
- 添加热噪声和少量生理噪声
- 目标SNR设置在15dB左右
- 包含轻微的扫描器漂移

**真实策略**（方法学研究）：
- 包含所有主要噪声成分
- 目标SNR设置在10dB左右
- 模拟真实扫描环境

### 六、质量控制要点

**必检项目**：
1. **SNR水平**：确保在目标范围内
2. **频率分布**：检查是否存在过多低频成分
3. **空间分布**：验证噪声的空间合理性
4. **统计假设**：确认符合后续分析的前提条件

**警告信号**：
- SNR低于10dB或高于30dB
- 低频功率占比超过50%
- 噪声不符合正态分布
- 空间模式出现明显伪影

这种系统性的方法确保添加的噪声既符合fMRI数据的物理特性，又满足统计分析的要求，特别是FDR等多重比较校正方法的假设条件。



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
