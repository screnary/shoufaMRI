# -*- coding: utf-8 -*-
"""
噪声参数实验脚本
"""

import os,sys
import argparse
import numpy as np

# 快速路径设置
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)  # code/
sys.path.insert(0, parent_dir)

if __package__ is None:
    # 根据文件位置动态设置包名
    relative_path = os.path.relpath(current_dir, parent_dir)
    __package__ = relative_path.replace(os.sep, '.')
    print(f"设置包名: {__package__}")

from shoufaMRI.core.exp_frame_noise import NoiseExperimentRunner
from shoufaMRI.utils.config_loader import (
    load_experiment_config, create_sample_configs, 
    ConfigLoader, print_coordinate_info
)
import pdb

# print("import success")
# pdb.set_trace()

def main():
    """主函数"""
    
    parser = argparse.ArgumentParser(description="噪声参数实验脚本")
    parser.add_argument("--config", "-c", default="./config/config_20251104_pre.yaml", 
                       help="配置文件路径 (默认: ./config/config_20250730.yaml)")
    parser.add_argument("--create-config", action="store_true",
                       help="创建示例配置文件")
    parser.add_argument("--coordinate-group", "-g", default="whole",
                       help="使用的坐标组名称 (默认: whole)")
    parser.add_argument("--list-coordinates", action="store_true",
                       help="列出配置文件中的所有坐标组")
    parser.add_argument("--experiment-name", "-n", default="[20251103] for original data",
                       help="实验名称 (覆盖配置文件中的设置)")
    parser.add_argument("--dry-run", action="store_true",
                       help="仅显示配置信息，不执行实验")
    
    args = parser.parse_args()
    
    # 创建示例配置文件
    if args.create_config:
        create_sample_configs()
        return
    
    # 列出坐标组
    if args.list_coordinates:
        print(args)
        print_coordinate_info(args.config)
        return
    
    try:
        # 加载配置
        print(f"加载配置文件: {args.config}")
        print(f"使用坐标组: {args.coordinate_group}")
        
        config = load_experiment_config(args.config, args.coordinate_group)
        
        # 覆盖实验名称
        experiment_name = args.experiment_name or None
        
        if args.dry_run:
            # 干运行模式：仅显示配置信息
            print_config_summary(config, args.coordinate_group)
            return 0
        
        # 运行实验
        print(f"开始实验...")
        runner = NoiseExperimentRunner(config, experiment_name)
        results = runner.run_experiment()
        
        # 显示结果摘要
        print_results_summary(results)
        
    except Exception as e:
        print(f"实验失败: {e}")
        return 1
    
    return 0

def print_config_summary(config, coordinate_group: str):
    """打印配置摘要"""
    
    print("\n" + "="*60)
    print("配置摘要")
    print("="*60)
    
    print(f"数据根目录: {config.data_root}")
    print(f"输入子目录: {config.input_subdir}")
    print(f"输出基础目录: {config.output_base_subdir}")
    
    print(f"\n使用的坐标组: {coordinate_group}")
    print(f"MNI坐标点数: {len(config.mni_coordinates)}")
    
    # 显示坐标点（最多显示前10个）
    print(f"坐标点列表:")
    for i, coord in enumerate(config.mni_coordinates[:10]):
        print(f"  {i+1:2d}. ({coord[0]:6.1f}, {coord[1]:6.1f}, {coord[2]:6.1f})")
    
    if len(config.mni_coordinates) > 10:
        print(f"  ... 还有 {len(config.mni_coordinates)-10} 个坐标点")
    
    print(f"\n避开半径 ({len(config.avoid_radius_list)} 个): {config.avoid_radius_list}")
    
    print(f"\n噪声参数 ({len(config.noise_params_list)} 个):")
    for i, params in enumerate(config.noise_params_list):
        print(f"  {i+1}. {params}")
    
    # 计算总任务数
    try:
        from utils import directory_utils
        nii_dir = os.path.join(config.data_root, config.input_subdir)
        if os.path.exists(nii_dir):
            nii_files = directory_utils.get_nii_files_with_pattern(nii_dir)
            total_files = len(nii_files)
        else:
            total_files = "未知(目录不存在)"
    except:
        total_files = "未知"
    
    total_combinations = len(config.avoid_radius_list) * len(config.noise_params_list)
    
    print(f"\n任务统计:")
    print(f"  NIfTI文件数: {total_files}")
    print(f"  参数组合数: {total_combinations}")
    if isinstance(total_files, int):
        print(f"  总任务数: {total_files * total_combinations}")
        
        # 估算时间
        avg_time_per_task = 30  # 秒，估算值
        total_estimated_time = total_files * total_combinations * avg_time_per_task
        print(f"  估算总时间: {total_estimated_time/3600:.1f} 小时")
    
    print(f"\n执行选项:")
    print(f"  最大并行进程: {config.max_workers}")
    print(f"  跳过已存在文件: {config.skip_existing}")
    print(f"  保存中间结果: {config.save_intermediate}")
    print(f"  最大处理样本量: {config.max_subjects}")
    
    print("="*60)

def print_results_summary(results):
    """打印结果摘要"""
    
    if not results:
        print("没有结果数据")
        return
    
    successful = [r for r in results if r.success]
    failed = [r for r in results if not r.success]
    
    print("\n" + "="*60)
    print("实验结果摘要")
    print("="*60)
    
    print(f"总任务数: {len(results)}")
    print(f"成功任务: {len(successful)} ({len(successful)/len(results)*100:.1f}%)")
    print(f"失败任务: {len(failed)} ({len(failed)/len(results)*100:.1f}%)")
    
    if successful:
        times = [r.processing_time for r in successful]
        print(f"\n处理时间统计:")
        print(f"  总时间: {sum(times):.2f} 秒 ({sum(times)/3600:.2f} 小时)")
        print(f"  平均时间: {np.mean(times):.2f} 秒/任务")
        print(f"  最短时间: {min(times):.2f} 秒")
        print(f"  最长时间: {max(times):.2f} 秒")
    
    if failed:
        print(f"\n失败任务详情:")
        for i, result in enumerate(failed[:5]):  # 只显示前5个
            print(f"  {i+1}. {os.path.basename(result.input_file)}: {result.error_message}")
        if len(failed) > 5:
            print(f"  ... 还有 {len(failed)-5} 个失败任务")
    
    print("="*60)


if __name__ == "__main__":
    exit(main())  # echo $?   # show return code
