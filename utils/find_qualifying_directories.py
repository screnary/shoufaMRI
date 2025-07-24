import os
import re
import argparse
import glob
import pdb

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


def main():
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
    matching_dirs = main()
