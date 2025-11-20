import os
import pandas as pd
import shutil
from collections import defaultdict
import pdb

def extract_mri_by_cam_category(excel_file, sheet_name="剔除无MRI"):
    """
    Step 1: 从Excel中按照CAM评分术后分类，提取MRI排序列表
    
    参数:
        excel_file: Excel文件路径
        sheet_name: sheet名称
    
    返回:
        cam_categories: 字典 {CAM类别: [MRI排序列表]}
    """
    print("=" * 60)
    print("Step 1: 从Excel提取CAM评分分类")
    print("=" * 60)
    
    # 读取Excel文件
    df = pd.read_excel(excel_file, sheet_name=sheet_name)
    
    # 清理列名（去除空格）
    df.columns = df.columns.str.strip()
    
    print(f"读取数据: {len(df)} 行")
    print(f"列名: {df.columns.tolist()}")
    
    # 检查必要的列是否存在
    if 'CAM评分术后' not in df.columns or 'MRI排序' not in df.columns:
        print(f"错误: 未找到必要的列")
        print(f"当前列名: {df.columns.tolist()}")
        return None
    
    # 按CAM评分分类
    cam_categories = defaultdict(list)
    
    for idx, row in df.iterrows():
        cam_score = row['CAM评分术后']
        mri_order = row['MRI排序']
        
        # 跳过空值
        if pd.isna(cam_score) or pd.isna(mri_order):
            continue
        
        # 清理数据（去除空格，转换为字符串）
        cam_score = str(cam_score).strip()
        mri_order = str(mri_order).strip()
        
        # 添加到对应类别
        cam_categories[cam_score].append(mri_order)
    
    # 打印统计信息
    print(f"\n找到 {len(cam_categories)} 个CAM评分类别:")
    for category, mri_list in sorted(cam_categories.items()):
        print(f"  - {category}: {len(mri_list)} 个样本")
        print(f"    样本: {mri_list[:5]}{'...' if len(mri_list) > 5 else ''}")
    
    return dict(cam_categories)


def copy_files_by_cam_category(source_dir, output_base_dir, cam_categories, 
                                category_name_mapping=None):
    """
    Step 2: 按照CAM评分分类，将文件拷贝到对应文件夹
    
    参数:
        source_dir: 源文件夹路径（包含zsub_****_submatrix.txt文件）
        output_base_dir: 输出基础目录
        cam_categories: CAM分类字典 {CAM类别: [MRI排序列表]}
        category_name_mapping: 类别名称映射 {中文名: 英文名}，如果为None则使用默认映射
    
    返回:
        copy_stats: 拷贝统计信息
    """
    print("\n" + "=" * 60)
    print("Step 2: 按CAM分类拷贝文件")
    print("=" * 60)
    
    # 默认类别名称映射
    if category_name_mapping is None:
        category_name_mapping = {
            '正常': 'normal',
            '谵妄': 'delirium',
            '不全对': 'incomplete',
            '待数据': 'pending_data'
        }
    
    # 创建输出基础目录
    if not os.path.exists(output_base_dir):
        os.makedirs(output_base_dir)
        print(f"创建输出目录: {output_base_dir}")
    
    # 统计信息
    copy_stats = defaultdict(lambda: {'found': 0, 'copied': 0, 'missing': []})
    
    # 获取源目录中所有的submatrix文件
    source_files = {}
    for filename in os.listdir(source_dir):
        if filename.endswith('_submatrix.txt'):
            # 提取MRI编号，例如从 zsub_0001_submatrix.txt 提取 sub_0001
            # 或从 sub_0056_submatrix.txt 提取 sub_0056
            parts = filename.split('_submatrix')[0]
            
            # 处理可能的前缀 z
            if parts.startswith('zsub_'):
                mri_id = 'sub_' + parts.split('zsub_')[1]
            elif parts.startswith('sub_'):
                mri_id = parts
            else:
                continue
            
            # 转换为大小写不敏感的格式（统一为小写）
            mri_id_lower = mri_id.lower()
            source_files[mri_id_lower] = filename
    
    print(f"\n源目录中找到 {len(source_files)} 个submatrix文件")
    
    # 按类别处理
    for cam_category, mri_list in cam_categories.items():
        # 获取输出目录名称
        output_dir_name = category_name_mapping.get(cam_category, cam_category)
        output_dir = os.path.join(output_base_dir, output_dir_name)
        
        # 创建类别目录
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
            print(f"\n创建类别目录: {output_dir}")
        
        print(f"\n处理类别: {cam_category} -> {output_dir_name}")
        print(f"  需要拷贝 {len(mri_list)} 个文件")
        
        # 处理该类别下的每个MRI文件
        for mri_order in mri_list:
            # 清理MRI编号（统一为小写）
            mri_order_clean = mri_order.strip().lower()
            
            # 在源文件中查找匹配的文件
            source_filename = None
            if mri_order_clean in source_files:
                source_filename = source_files[mri_order_clean]
            else:
                # 尝试其他可能的格式
                for key in source_files.keys():
                    if mri_order_clean in key or key in mri_order_clean:
                        source_filename = source_files[key]
                        break
            
            if source_filename:
                # 找到文件，执行拷贝
                source_path = os.path.join(source_dir, source_filename)
                dest_path = os.path.join(output_dir, source_filename)
                
                try:
                    shutil.copy2(source_path, dest_path)
                    copy_stats[cam_category]['copied'] += 1
                    copy_stats[cam_category]['found'] += 1
                    print(f"  ✓ 拷贝: {source_filename}")
                except Exception as e:
                    print(f"  ✗ 拷贝失败 {source_filename}: {str(e)}")
            else:
                # 未找到文件
                copy_stats[cam_category]['missing'].append(mri_order)
                print(f"  ✗ 未找到文件: {mri_order}")
    
    # 打印统计信息
    print("\n" + "=" * 60)
    print("拷贝统计信息")
    print("=" * 60)
    
    for category in sorted(copy_stats.keys()):
        stats = copy_stats[category]
        output_name = category_name_mapping.get(category, category)
        print(f"\n{category} ({output_name}):")
        print(f"  需要文件数: {stats['found'] + len(stats['missing'])}")
        print(f"  成功拷贝: {stats['copied']}")
        print(f"  未找到: {len(stats['missing'])}")
        
        if stats['missing']:
            print(f"  缺失列表: {stats['missing'][:10]}")
            if len(stats['missing']) > 10:
                print(f"    ... 还有 {len(stats['missing']) - 10} 个")
    
    return dict(copy_stats)


def main_process(excel_file, sheet_name, source_dir, output_base_dir, 
                 category_name_mapping=None):
    """
    主处理流程
    
    参数:
        excel_file: Excel文件路径
        sheet_name: sheet名称
        source_dir: 源文件目录
        output_base_dir: 输出基础目录
        category_name_mapping: 类别名称映射
    """
    # Step 1: 提取CAM分类
    cam_categories = extract_mri_by_cam_category(excel_file, sheet_name)
    
    if cam_categories is None:
        print("提取CAM分类失败，程序终止")
        return
    
    # Step 2: 拷贝文件
    copy_stats = copy_files_by_cam_category(
        source_dir, 
        output_base_dir, 
        cam_categories,
        category_name_mapping
    )
    
    print("\n" + "=" * 60)
    print("处理完成！")
    print("=" * 60)
    
    return cam_categories, copy_stats


# 使用示例
if __name__ == "__main__":
    # 配置参数
    data_dir = "/mnt/c/Works/ws/shoufa2025/data/matrix"
    
    # Excel文件路径
    excel_file = os.path.join(data_dir, "首发入组MRI对照表.xlsx")
    sheet_name = "剔除无MRI"
    
    # 源文件目录（包含submatrix文件）
    source_dir = os.path.join(data_dir, "processed/post")
    
    # 输出基础目录
    output_base_dir = os.path.join(data_dir, "processed_classified_by_cam/post")
    
    # 类别名称映射（中文到英文）
    category_name_mapping = {
        '正常': 'normal',
        '谵妄': 'delirium',
        '不全对': 'incomplete',
        '待数据': 'pending_data',
        '患者出院就算正常吧哈哈哈哈': 'normal'
    }
    
    try:
        cam_categories, copy_stats = main_process(
            excel_file=excel_file,
            sheet_name=sheet_name,
            source_dir=source_dir,
            output_base_dir=output_base_dir,
            category_name_mapping=category_name_mapping
        )
        
        # 可选：保存分类结果到文件
        import json
        
        # 保存CAM分类列表
        output_json = os.path.join(output_base_dir, "cam_categories.json")
        with open(output_json, 'w', encoding='utf-8') as f:
            json.dump(cam_categories, f, ensure_ascii=False, indent=2)
        print(f"\nCAM分类列表已保存到: {output_json}")
        
        # 保存拷贝统计
        stats_json = os.path.join(output_base_dir, "copy_statistics.json")
        with open(stats_json, 'w', encoding='utf-8') as f:
            json.dump(copy_stats, f, ensure_ascii=False, indent=2)
        print(f"拷贝统计已保存到: {stats_json}")
        
    except Exception as e:
        print(f"程序执行出错: {str(e)}")
        import traceback
        traceback.print_exc()
