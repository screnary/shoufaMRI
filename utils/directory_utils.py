"""
all directory utilities
"""

import os
import re
import argparse
from pathlib import Path
import glob
import shutil

import pdb


def get_nii_files_with_pattern(directory, pattern="*.nii*", recursive=True):
    """
    获取目录下所有.nii文件的完整路径
    
    Parameters:
    -----------
    directory : str
        顶级目录路径
    pattern : str, default="*.nii*"
        文件匹配模式，支持 .nii, .nii.gz 等
    recursive : bool, default=True
        是否递归搜索子目录
    
    Returns:
    --------
    list
        包含所有.nii文件完整路径的列表
    
    Examples:
    ---------
    目录结构:
    data/
    ├── subject_001/
    │   └── brain.nii.gz
    ├── subject_002/
    │   └── brain.nii
    └── subject_003/
        └── scan.nii.gz
    
    >>> files = get_nii_files('data/')
    >>> print(files)
    ['data/subject_001/brain.nii.gz', 'data/subject_002/brain.nii', 'data/subject_003/scan.nii.gz']
    """
    
    matched_files = []
    directory = Path(directory)
    
    if not directory.exists():
        print(f"警告: 目录 {directory} 不存在")
        return []
    
    if recursive:
        # 方法1: 使用pathlib递归搜索 (推荐)
        matched_files = list(directory.rglob(pattern))
    else:
        # 只搜索直接子目录
        matched_files = list(directory.glob(pattern))
    
    # 转换为字符串路径并排序
    matched_files = sorted([str(f) for f in matched_files])
    # 只保留.nii或.nii.gz文件
    matched_files = [f for f in matched_files 
                     if f.suffix in ['.nii', '.gz'] and str(f).endswith(('.nii', '.nii.gz'))]
    
    return matched_files


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
    3. 目录中包含12000个.dcm文件
    
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
        
        # 检查3: 是否包含12000个.dcm文件
        dcm_files = glob.glob(os.path.join(dirpath, '*.dcm'))
        dcm_count = len(dcm_files)
        
        if dcm_count == 12000:
            qualifying_dirs.append(os.path.abspath(dirpath))
    #  print(f"找到符合条件的目录: {os.path.abspath(dirpath)}")
    
    return sorted(qualifying_dirs, key=extract_subject_number)


def copy_files_with_pattern_from_subdirs(source_dir, target_dir, pattern="bcNGS*.nii*", recursive_search=False):
    """
    从源目录的所有sub_*子目录中提取匹配特定模式的.nii文件，
    复制到目标目录下对应的sub_*子目录中
    
    Parameters:
    -----------
    source_dir : str
        源目录路径，如 E:\data\liuyang\original_202510\for_original\Rest_pre\
    target_dir : str
        目标目录路径，如 E:\data\liuyang\original_202510\for_original_bcNGS\Rest_pre\
    pattern : str, default="bcNGS*.nii*"
        文件匹配模式，支持通配符
        - "bcNGS*.nii*" : 匹配bcNGS开头的文件
        - "2024*.nii*" : 匹配2024开头的文件  
        - "[0-9]*.nii*" : 匹配数字开头的文件（需要用glob模式）
    recursive_search : bool, default=False
        是否在每个sub_*子目录中递归搜索
    
    Returns:
    --------
    dict
        包含统计信息的字典：{'total_subjects': int, 'total_files': int, 'failed': list}
    
    Examples:
    ---------
    # 提取bcNGS前缀的文件
    >>> stats = copy_files_with_pattern_from_subdirs(
    ...     'E:\\data\\liuyang\\original_202510\\for_original\\Rest_pre\\',
    ...     'E:\\data\\liuyang\\original_202510\\for_original_bcNGS\\Rest_pre\\',
    ...     pattern="bcNGS*.nii*"
    ... )
    
    # 提取数字开头的文件
    >>> stats = copy_files_with_pattern_from_subdirs(
    ...     'E:\\data\\liuyang\\original_202510\\for_original\\Rest_pre\\',
    ...     'E:\\data\\liuyang\\original_202510\\for_original_numbers\\Rest_pre\\',
    ...     pattern="[0-9]*.nii*"
    ... )
    """
    source_path = Path(source_dir)
    target_path = Path(target_dir)
    
    # 检查源目录是否存在
    if not source_path.exists():
        print(f"错误: 源目录 {source_path} 不存在")
        return {'total_subjects': 0, 'total_files': 0, 'failed': []}
    
    # 创建目标基础目录
    target_path.mkdir(parents=True, exist_ok=True)
    
    # 统计信息
    stats = {
        'total_subjects': 0,
        'total_files': 0,
        'failed': []
    }
    
    # 遍历所有sub_*子目录
    sub_dirs = sorted([d for d in source_path.iterdir() 
                      if d.is_dir() and d.name.startswith('sub_')],
                     key=lambda x: extract_subject_number(str(x)))
    
    if not sub_dirs:
        print(f"警告: 在 {source_path} 中未找到sub_*子目录")
        return stats
    
    print(f"找到 {len(sub_dirs)} 个被试目录")
    print(f"匹配模式: {pattern}")
    print(f"递归搜索: {'是' if recursive_search else '否'}")
    print(f"开始处理...\n")
    
    # 处理每个被试目录
    for sub_dir in sub_dirs:
        stats['total_subjects'] += 1
        sub_name = sub_dir.name
        
        # 使用get_nii_files_with_pattern函数查找匹配的文件
        matched_files = get_nii_files_with_pattern(
            sub_dir, 
            pattern, 
            recursive=recursive_search
        )
        
        if not matched_files:
            print(f"[{sub_name}] 未找到匹配 '{pattern}' 的文件")
            continue
        
        # 创建对应的目标子目录
        target_sub_dir = target_path / sub_name
        target_sub_dir.mkdir(parents=True, exist_ok=True)
        
        # 复制文件
        copied_count = 0
        for file_path in matched_files:
            try:
                target_file = target_sub_dir / file_path.name
                shutil.copy2(file_path, target_file)
                copied_count += 1
                stats['total_files'] += 1
            except Exception as e:
                error_msg = f"{sub_name}/{file_path.name}: {str(e)}"
                stats['failed'].append(error_msg)
                print(f"  错误: {error_msg}")
        
        print(f"[{sub_name}] 复制了 {copied_count} 个文件")
    
    # 打印总结
    print(f"\n{'='*60}")
    print(f"处理完成！")
    print(f"  处理被试数: {stats['total_subjects']}")
    print(f"  复制文件数: {stats['total_files']}")
    if stats['failed']:
        print(f"  失败数量: {len(stats['failed'])}")
        print(f"  失败详情:")
        for fail in stats['failed']:
            print(f"    - {fail}")
    print(f"  目标目录: {target_path}")
    print(f"{'='*60}")
    
    return stats


def copy_numeric_prefix_files(source_dir, target_dir, recursive_search=False):
    """
    便捷函数：专门用于提取数字开头的.nii文件
    如：20240416_072804LIUYANGFMRIs002a1001.nii
    
    Parameters:
    -----------
    source_dir : str
        源目录路径
    target_dir : str
        目标目录路径
    recursive_search : bool
        是否递归搜索
    """
    return copy_files_with_pattern_from_subdirs(
        source_dir, 
        target_dir, 
        pattern="[0-9]*.nii*",
        recursive_search=recursive_search
    )


def copy_bcNGS_files(source_dir, target_dir, recursive_search=False):
    """
    便捷函数：专门用于提取bcNGS前缀的.nii文件
    
    Parameters:
    -----------
    source_dir : str
        源目录路径
    target_dir : str
        目标目录路径
    recursive_search : bool
        是否递归搜索
    """
    return copy_files_with_pattern_from_subdirs(
        source_dir, 
        target_dir, 
        pattern="bcNGS*.nii*",
        recursive_search=recursive_search
    )


# 更新命令行入口
def main_copy_files():
    """用于复制特定模式文件的命令行入口"""
    parser = argparse.ArgumentParser(
        description='从子目录复制匹配特定模式的.nii文件',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 复制bcNGS前缀的文件
  python directory_utils.py --copy source_dir target_dir --pattern "bcNGS*.nii*"
  
  # 复制数字开头的文件
  python directory_utils.py --copy source_dir target_dir --pattern "[0-9]*.nii*"
  
  # 复制2024开头的文件并递归搜索
  python directory_utils.py --copy source_dir target_dir --pattern "2024*.nii*" --recursive
        """
    )
    parser.add_argument('source_dir', help='源目录路径（包含sub_*子目录）')
    parser.add_argument('target_dir', help='目标目录路径')
    parser.add_argument('--pattern', '-p', 
                       default='bcNGS*.nii*',
                       help='文件匹配模式（默认: bcNGS*.nii*）')
    parser.add_argument('--recursive', '-r', 
                       action='store_true',
                       help='在每个sub_*子目录中递归搜索')
    
    args = parser.parse_args()
    
    stats = copy_files_with_pattern_from_subdirs(
        args.source_dir, 
        args.target_dir,
        pattern=args.pattern,
        recursive_search=args.recursive
    )
    return stats


def main_test_find_dirs():
    parser = argparse.ArgumentParser(description='找出符合特定条件的目录')
    parser.add_argument('directory', help='要搜索的根目录路径')
    parser.add_argument('--output', '-o', help='结果输出文件路径（可选）')
    
    args = parser.parse_args()
    
    root_directory = args.directory
    if not os.path.isdir(root_directory):
        print(f"错误: '{root_directory}' 不是一个有效的目录")
        return
    
    print(f"开始在 '{root_directory}' 中搜索符合条件的目录...")
    qualifying_dirs = find_qualifying_directories(root_directory)
    
    print(f"\n搜索完成，共找到 {len(qualifying_dirs)} 个符合条件的目录")
    
    if args.output:
        with open(args.output, 'w', encoding='utf-8') as f:
            for dir_path in qualifying_dirs:
                f.write(f"{dir_path}\n")
        print(f"结果已保存到: {args.output}")
    
    return qualifying_dirs


if __name__ == "__main__":
    # matching_dirs = main_test_find_dirs()
    main_copy_files()  # python directory_utils.py --copy source_dir target_dir --pattern "bcNGS*.nii*"
