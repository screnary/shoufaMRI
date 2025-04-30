import os
import shutil
import re
import argparse

def organize_files_by_parity(directory):
    """
    将目录中按数字命名的文件分类到奇数(odd)和偶数(even)文件夹中
    
    Args:
        directory: 要处理的目录路径
    """
    # 确保目录存在
    if not os.path.isdir(directory):
        print(f"错误: '{directory}' 不是一个有效的目录")
        return False
    
    # 创建 odd 和 even 子目录
    odd_dir = os.path.join(directory, "odd")
    even_dir = os.path.join(directory, "even")
    
    # 如果子目录不存在，则创建它们
    os.makedirs(odd_dir, exist_ok=True)
    os.makedirs(even_dir, exist_ok=True)
    
    # 用于匹配数字文件名的模式
    number_pattern = re.compile(r'^(\d+)(\..+)?$')
    
    # 计数器
    count_odd = 0
    count_even = 0
    
    # 遍历目录中的所有文件
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
            
            # 确定奇偶性并移动文件
            if file_number % 2 == 0:  # 偶数
                destination = os.path.join(even_dir, filename)
                shutil.copy2(file_path, destination)
                count_even += 1
            else:  # 奇数
                destination = os.path.join(odd_dir, filename)
                shutil.copy2(file_path, destination)
                count_odd += 1
    
    print(f"处理完成:")
    print(f"- 移动了 {count_odd} 个文件到奇数目录 (odd)")
    print(f"- 移动了 {count_even} 个文件到偶数目录 (even)")
    
    return True

if __name__ == "__main__":
    """
    Usage:
    python organize_by_parity.py /path/to/your/directory  # copy files
    python organize_by_parity.py /path/to/your/directory --move  # move files
    """
    # 创建命令行参数解析器
    parser = argparse.ArgumentParser(description="将数字命名的文件按奇偶性分类到不同文件夹")
    parser.add_argument("directory", help="要处理的目录路径")
    parser.add_argument("--move", action="store_true", help="移动文件而不是复制")
    
    # 解析命令行参数
    args = parser.parse_args()
    
    # 使用指定的移动方式
    if args.move:
        print(f"将从 '{args.directory}' 移动文件...")
        # 如果选择移动，替换复制函数
        shutil.copy2 = shutil.move
    else:
        print(f"将从 '{args.directory}' 复制文件...")
    
    # 执行文件分类
    organize_files_by_parity(args.directory)
