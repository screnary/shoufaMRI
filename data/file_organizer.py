import os
import shutil
import re
import argparse
import math

def organize_files_by_batch_parity(directory, out_directory, batch_size):
    """
    将目录中按数字命名的文件按每batch_size个为一堆，区分奇偶batch分别存放到不同文件夹
    
    Args:
        directory: 要处理的目录路径
        out_directory: where to store processed files
        batch_size: 每堆的文件数量
    
    Returns:
        bool: 是否成功处理
    """
    # 确保目录存在
    if not os.path.isdir(directory):
        print(f"错误: '{directory}' 不是一个有效的目录")
        return False
    
    # 创建 odd 和 even 子目录
    odd_dir = os.path.join(out_directory, "odd")
    even_dir = os.path.join(out_directory, "even")
    
    # 如果子目录不存在，则创建它们
    os.makedirs(odd_dir, exist_ok=True)
    os.makedirs(even_dir, exist_ok=True)
    
    # 用于匹配数字文件名的模式
    # number_pattern = re.compile(r'^(\d+)(\..+)?$')
    number_pattern = re.compile(r'^(\d+)\.dcm$')
    
    # 收集所有数字命名的文件
    numbered_files = []
    
    for filename in os.listdir(directory):
        file_path = os.path.join(directory, filename)
        
        # 只处理文件，跳过目录
        if not os.path.isfile(file_path):
            continue
            
        # 跳过 odd 和 even 目录本身
        if filename in ["odd", "even"]:
            continue
        
        # 检查文件名是否是数字
        match = number_pattern.match(filename)
        if match:
            # 提取数字部分
            file_number = int(match.group(1))
            numbered_files.append((file_number, filename, file_path))
    
    # 按文件名数字排序
    numbered_files.sort(key=lambda x: x[0])
    
    # 统计
    odd_batch_count = 0
    even_batch_count = 0
    odd_file_count = 0
    even_file_count = 0
    
    # 处理文件
    for i, (file_number, filename, file_path) in enumerate(numbered_files):
        # 计算文件所在的堆
        batch_number = math.ceil((i + 1) / batch_size)
        
        # 确定堆的奇偶性
        if batch_number % 2 == 0:  # 偶数堆
            destination = os.path.join(even_dir, filename)
            shutil.copy2(file_path, destination)
            even_file_count += 1
            if i % batch_size == 0 or i == 0:
                even_batch_count += 1
# print(f"开始处理偶数堆 #{batch_number}，范围: {i+1}~{min(i+batch_size, len(numbered_files))}")
        else:  # 奇数堆
            destination = os.path.join(odd_dir, filename)
            shutil.copy2(file_path, destination)
            odd_file_count += 1
            if i % batch_size == 0 or i == 0:
                odd_batch_count += 1
# print(f"开始处理奇数堆 #{batch_number}，范围: {i+1}~{min(i+batch_size, len(numbered_files))}")
    
    print(f"\n处理完成:")
    print(f"- 奇数堆: {odd_batch_count} 堆，共 {odd_file_count} 个文件")
    print(f"- 偶数堆: {even_batch_count} 堆，共 {even_file_count} 个文件")
    
    return True


def organize_files_by_batch_half(directory, out_directory, batch_size):
    """
    将目录中按数字命名的文件按每batch_size个为一堆，前半部分和后半部分分别存放到不同文件夹
    
    Args:
        directory: 要处理的目录路径
        out_directory: where to store processed files
        batch_size: 每堆的文件数量
    
    Returns:
        bool: 是否成功处理
    """
    # 确保目录存在
    if not os.path.isdir(directory):
        print(f"错误: '{directory}' 不是一个有效的目录")
        return False
    
    # 创建 first_half 和 second_half 子目录
    first_half_dir = os.path.join(out_directory, "first_half")
    second_half_dir = os.path.join(out_directory, "second_half")
    
    # 如果子目录不存在，则创建它们
    os.makedirs(first_half_dir, exist_ok=True)
    os.makedirs(second_half_dir, exist_ok=True)
    
    # 用于匹配数字文件名的模式
    # number_pattern = re.compile(r'^(\d+)(\..+)?$')
    number_pattern = re.compile(r'^(\d+)\.dcm$')
    # 收集所有数字命名的文件
    numbered_files = []
    
    for filename in os.listdir(directory):
        file_path = os.path.join(directory, filename)
        
        # 只处理文件，跳过目录
        if not os.path.isfile(file_path):
            continue
            
        # 跳过目录本身
        if filename in ["first_half", "second_half"]:
            continue
        
        # 检查文件名是否是数字
        match = number_pattern.match(filename)
        if match:
            # 提取数字部分
            file_number = int(match.group(1))
            numbered_files.append((file_number, filename, file_path))
    
    # 按文件名数字排序
    numbered_files.sort(key=lambda x: x[0])
    
    # 统计
    first_half_batch_count = 0
    second_half_batch_count = 0
    first_half_file_count = 0
    second_half_file_count = 0
    
    # 计算总批次数
    total_files = len(numbered_files)
    total_batches = math.ceil(total_files / batch_size)
    half_batch_point = total_batches // 2
    
    # 如果总批次数为奇数，确保前半部分多一个批次
    if total_batches % 2 == 1:
        half_batch_point += 1
    
    print(f"总文件数: {total_files}, 总批次数: {total_batches}, 分割点: 第{half_batch_point}批")
    
    # 处理文件
    for i, (file_number, filename, file_path) in enumerate(numbered_files):
        # 计算文件所在的堆编号 (从1开始)
        batch_number = math.ceil((i + 1) / batch_size)
        
        # 确定是前半部分还是后半部分
        if batch_number <= half_batch_point:  # 前半部分
            destination = os.path.join(first_half_dir, filename)
            shutil.copy2(file_path, destination)
            first_half_file_count += 1
            if i % batch_size == 0 or i == 0:
                first_half_batch_count += 1
# print(f"开始处理前半部分批次 #{batch_number}，范围: {i+1}~{min(i+batch_size, len(numbered_files))}")
        else:  # 后半部分
            destination = os.path.join(second_half_dir, filename)
            shutil.copy2(file_path, destination)
            second_half_file_count += 1
            if i % batch_size == 0:
                second_half_batch_count += 1
# print(f"开始处理后半部分批次 #{batch_number}，范围: {i+1}~{min(i+batch_size, len(numbered_files))}")
    
    print(f"\n处理完成:")
    print(f"- 前半部分: {first_half_batch_count} 批次，批次范围 1-{half_batch_point}，共 {first_half_file_count} 个文件")
    print(f"- 后半部分: {second_half_batch_count} 批次，批次范围 {half_batch_point+1}-{total_batches}，共 {second_half_file_count} 个文件")
    
    return True



def organize_files_by_batch(directory, out_directory, batch_size=40, mode='parity'):
    """
    将目录中按数字命名的文件按每batch_size个为一堆，根据堆的奇偶性分类
    
    Args:
        directory: 要处理的目录路径
        batch_size: 每堆的文件数量
        mode: ['parity', 'half'], 'parity'是奇偶样本划分，'half'是前后对半分
    """
    if mode == 'parity':
        organize_files_by_batch_parity(directory, out_directory, batch_size)
    elif mode == 'half':
        organize_files_by_batch_half(directory, out_directory, batch_size)


# 20251103: copy files to new directory

if __name__ == "__main__":
    """ USAGE
    python organize_files_by_batch_parity.py /mnt/c/Works/ws/shoufa2025/data/002_AxBOLD --move
    """
    # 创建命令行参数解析器
    parser = argparse.ArgumentParser(description="将数字命名的文件按堆的奇偶性分类到不同文件夹")
    parser.add_argument("directory", help="要处理的目录路径")
    parser.add_argument("--out_directory", "-o", dest="out_directory", help="目标文件要存放的路径(默认使用directory值)")
    parser.add_argument("--batch-size", type=int, default=40, help="每堆的文件数量 (默认: 40)")
    parser.add_argument("--mode", choices=["parity", "half"], default='parity', help="数据划分模式 (默认: 'parity'); select from ['parity', 'half']")
    parser.add_argument("--move", action="store_true", help="移动文件而不是复制")
    
    # 解析命令行参数
    args = parser.parse_args()

    # 如果未提供输出目录，则使用输入目录
    if args.out_directory is None:
        args.out_directory = args.directory
    
    # 使用指定的移动方式
    if args.move:
        print(f"将从 '{args.directory}' 移动文件...")
        # 如果选择移动，替换复制函数
        shutil.copy2 = shutil.move
    else:
        print(f"将从 '{args.directory}' 复制文件...")
    
    # 执行文件分类
    organize_files_by_batch(args.directory, args.out_directory, args.batch_size, args.mode)
