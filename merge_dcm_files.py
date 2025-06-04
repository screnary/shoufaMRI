"""
合并 AxBOLD 目录下所有子目录中的 .dcm 文件到一个新的目录

用法:
python merge_dcm_files.py /path/to/source/AxBOLD /path/to/destination/AxBOLD

该脚本会:
1. 递归查找源 AxBOLD 目录下所有 .dcm 文件
2. 将它们复制到目标目录
3. 如有重名文件，将添加序号确保唯一性
4. 输出合并统计信息
"""

import os
import re
import sys
import glob
import shutil
from pathlib import Path
import logging

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


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
    - 2. 为最末级子目录（不包含子目录）[depicted] -
    - 3. 目录中包含12000个.dcm文件 -
    
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
        
        print(f"找到符合条件的目录: {os.path.abspath(dirpath)}")
        qualifying_dirs.append(os.path.abspath(dirpath))
    
    return sorted(qualifying_dirs, key=extract_subject_number)


def merge_dcm_files(source_dir, dest_dir):
    """
    合并源目录下所有子目录中的 .dcm 文件到目标目录
    
    Args:
        source_dir (str): 源 AxBOLD 目录路径
        dest_dir (str): 目标 AxBOLD 目录路径
    
    Returns:
        tuple: (成功复制的文件数, 总文件数)
    """
    # 将路径转为 Path 对象，便于处理
    source_path = Path(source_dir)
    dest_path = Path(dest_dir)
    
    # 检查源目录是否存在
    if not source_path.exists() or not source_path.is_dir():
        logger.error(f"源目录不存在或不是一个目录: {source_dir}")
        return 0, 0
    
    # 创建目标目录（如果不存在）
    os.makedirs(dest_path, exist_ok=True)
    logger.info(f"目标目录: {dest_path}")
    
    # 用于跟踪已使用的文件名和统计信息
    copied_files = 0
    total_files = 0
    existing_filenames = set()
    
    # 递归查找所有 .dcm 文件
    for root, _, files in os.walk(source_path):
        dcm_files = [f for f in files if f.lower().endswith('.dcm')]
        total_files += len(dcm_files)
        
        if dcm_files:
            rel_path = os.path.relpath(root, source_path)
            logger.info(f"正在处理子目录: {rel_path} (找到 {len(dcm_files)} 个 .dcm 文件)")
            
            for dcm_file in dcm_files:
                source_file = os.path.join(root, dcm_file)
                
                # 处理可能的文件名冲突
                if dcm_file in existing_filenames:
                    # 如果文件名已存在，添加序号
                    base_name, ext = os.path.splitext(dcm_file)
                    counter = 1
                    new_filename = f"{base_name}_{counter}{ext}"
                    
                    # 持续增加计数器直到找到唯一名称
                    while new_filename in existing_filenames:
                        counter += 1
                        new_filename = f"{base_name}_{counter}{ext}"
                    
                    dest_file = dest_path / new_filename
                    logger.debug(f"重命名: {dcm_file} -> {new_filename}")
                else:
                    dest_file = dest_path / dcm_file
                
                # 记录文件名，防止重复
                existing_filenames.add(dest_file.name)
                
                # 复制文件
                try:
                    shutil.copy2(source_file, dest_file)
                    copied_files += 1
                    
                    # 每复制100个文件输出一次进度
                    if copied_files % 100 == 0:
                        logger.info(f"已复制 {copied_files} 个文件...")
                        
                except Exception as e:
                    logger.error(f"复制文件失败: {source_file} -> {dest_file}")
                    logger.error(f"错误信息: {str(e)}")
    
    return copied_files, total_files

def main():
    """主函数"""
    # 检查命令行参数
    if len(sys.argv) != 3:
        print("用法: python merge_dcm_files.py 源AxBOLD目录 目标AxBOLD目录")
        sys.exit(1)
    
    source_dir = sys.argv[1]
    dest_dir = sys.argv[2]
    
    logger.info("开始合并 DICOM 文件...")
    logger.info(f"源目录: {source_dir}")
    
    # 执行合并
    start_time = os.times()
    copied_files, total_files = merge_dcm_files(source_dir, dest_dir)
    end_time = os.times()
    
    # 计算处理时间
    processing_time = end_time.user - start_time.user + end_time.system - start_time.system
    
    # 输出统计信息
    logger.info("=" * 50)
    logger.info("合并完成！")
    logger.info(f"找到的 .dcm 文件总数: {total_files}")
    logger.info(f"成功复制的文件数: {copied_files}")
    
    if total_files > 0:
        success_rate = (copied_files / total_files) * 100
        logger.info(f"成功率: {success_rate:.2f}%")
    
    logger.info(f"处理时间: {processing_time:.2f} 秒")
    logger.info("=" * 50)

if __name__ == "__main__":

    root_directory = '/mnt/e/data/liuyang/processed_202505/CFH_Original_Data'
    print(f"开始在 '{root_directory}' 中搜索符合条件的目录...")
    src_dir_list = find_qualifying_directories(root_directory)
    tar_dir_list = [path.replace('processed_202505', 'original_202505') for path in src_dir_list]
    total_dirs = len(src_dir_list)
    print(f"\n搜索完成，共找到 {total_dirs} 个符合条件的目录")
    
    for i,src_dir in enumerate(src_dir_list):
        tar_dir = tar_dir_list[i]
        print(f"将从 '{src_dir}' 复制文件到 '{tar_dir}'...")
    
        logger.info("开始合并 DICOM 文件...")
        logger.info(f"源目录: {src_dir}")
        
        # 执行合并
        start_time = os.times()
        copied_files, total_files = merge_dcm_files(src_dir, tar_dir)
        end_time = os.times()
        
        # 计算处理时间
        processing_time = end_time.user - start_time.user + end_time.system - start_time.system
        
        # 输出统计信息
        logger.info("=" * 50)
        logger.info("合并完成！")
        logger.info(f"找到的 .dcm 文件总数: {total_files}")
        logger.info(f"成功复制的文件数: {copied_files}")
        
        if total_files > 0:
            success_rate = (copied_files / total_files) * 100
            logger.info(f"成功率: {success_rate:.2f}%")
        
        logger.info(f"处理时间: {processing_time:.2f} 秒")
        logger.info("=" * 50)
