# -*- coding: utf-8 -*-
"""
可复用的噪声参数实验框架
用于批量测试不同的 noise_params 和 avoid_radius 组合
"""

import os
import sys
import json
import time
import logging
import itertools
from datetime import datetime
from typing import Dict, List, Tuple, Optional, Union
from dataclasses import dataclass, asdict
from pathlib import Path
import pandas as pd
import numpy as np
from concurrent.futures import ProcessPoolExecutor, as_completed
import traceback
import pdb

# 检查是否作为包的一部分运行
if __package__ is None:
    # 直接运行的情况，添加项目根目录到路径: (../../shoufaMRI)
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if project_root not in sys.path:
        sys.path.insert(0, project_root)
    parent_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    if parent_dir not in sys.path:
        sys.path.insert(0, parent_dir)
    
    # 设置包名
    __package__ = 'shoufaMRI.core'

# import custom module
from ..utils import directory_utils
from . import volume_noise_operations as noise


@dataclass
class ExperimentConfig:
    """实验配置数据类"""
    
    # 数据路径配置
    data_root: str
    input_subdir: str
    output_base_subdir: str
    
    # MNI坐标
    mni_coordinates: List[List[float]]
    
    # 参数范围
    avoid_radius_list: List[float]
    noise_params_list: List[Optional[Dict]]
    
    # 处理选项
    noise_type: str = 'gaussian'
    save_mask: bool = False
    use_brain_mask: bool = True
    
    # 实验选项
    max_workers: int = 4
    max_subjects: Optional[int] = None  # if assign subject number being processed
    skip_existing: bool = True
    save_intermediate: bool = True

@dataclass
class ExperimentResult:
    """单次实验结果"""
    
    experiment_id: str
    input_file: str
    output_file: str
    avoid_radius: float
    noise_params: Optional[Dict]
    
    # 结果统计
    processing_time: float
    success: bool
    error_message: Optional[str] = None
    
    # 数据统计（可选）
    original_stats: Optional[Dict] = None
    noisy_stats: Optional[Dict] = None
    mask_stats: Optional[Dict] = None

class NoiseExperimentRunner:
    """噪声参数实验运行器"""
    
    def __init__(self, config: ExperimentConfig, experiment_name: str = None):
        self.config = config
        self.experiment_name = experiment_name or f"noise_exp_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        # 创建实验目录
        self.experiment_dir = os.path.join(config.data_root, "experiments", self.experiment_name)
        os.makedirs(self.experiment_dir, exist_ok=True)
        
        # 设置日志
        self._setup_logging()
        
        # 保存配置
        self._save_config()
        
        # 结果存储
        self.results: List[ExperimentResult] = []
        
    def _setup_logging(self):
        """设置日志"""
        
        log_file = os.path.join(self.experiment_dir, "experiment.log")
        
        # 创建logger
        self.logger = logging.getLogger(self.experiment_name)
        self.logger.setLevel(logging.INFO)
        
        # 文件处理器
        file_handler = logging.FileHandler(log_file, encoding='utf-8')
        file_handler.setLevel(logging.INFO)
        
        # 控制台处理器
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        
        # 格式化
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        file_handler.setFormatter(formatter)
        console_handler.setFormatter(formatter)
        
        # 添加处理器
        self.logger.addHandler(file_handler)
        self.logger.addHandler(console_handler)
        
    def _save_config(self):
        """保存实验配置"""
        
        config_file = os.path.join(self.experiment_dir, "config.json")
        
        with open(config_file, 'w', encoding='utf-8') as f:
            json.dump(asdict(self.config), f, indent=2, ensure_ascii=False)
            
        self.logger.info(f"实验配置已保存: {config_file}")
    
    def _generate_parameter_combinations(self) -> List[Tuple[float, Optional[Dict]]]:
        """生成参数组合"""
        
        combinations = list(itertools.product(
            self.config.avoid_radius_list,
            self.config.noise_params_list
        ))
        
        self.logger.info(f"生成 {len(combinations)} 个参数组合")
        
        return combinations
    
    def _create_output_path(self, input_path: str, avoid_radius: float, 
                          noise_params: Optional[Dict]) -> str:
        """创建输出路径"""
        
        # 创建参数标识
        radius_str = f"r{avoid_radius:.1f}"
        
        if noise_params is None:
            noise_str = "default"
        else:
            # 从noise_params创建简短标识
            noise_parts = []
            for key, value in noise_params.items():
                if isinstance(value, float):
                    noise_parts.append(f"{key}{value:.1f}")
                else:
                    noise_parts.append(f"{key}{value}")
            noise_str = "_".join(noise_parts)
        
        # 创建输出子目录名
        output_subdir = f"{self.config.output_base_subdir}_{radius_str}_{noise_str}"
        
        # 替换路径
        output_path = input_path.replace(
            f"/{self.config.input_subdir}/",
            f"/{output_subdir}/"
        )
        
        # 确保输出目录存在
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        return output_path
    
    def _process_single_file(self, input_file: str, avoid_radius: float, 
                           noise_params: Optional[Dict]) -> ExperimentResult:
        """处理单个文件"""
        
        # 创建实验ID
        experiment_id = f"r{avoid_radius:.1f}_" + (
            "default" if noise_params is None else 
            "_".join([f"{k}{v}" for k, v in noise_params.items()])
        )
        
        # 创建输出路径
        output_file = self._create_output_path(input_file, avoid_radius, noise_params)
        
        # 检查是否跳过已存在的文件
        if self.config.skip_existing and os.path.exists(output_file):
            return ExperimentResult(
                experiment_id=experiment_id,
                input_file=input_file,
                output_file=output_file,
                avoid_radius=avoid_radius,
                noise_params=noise_params,
                processing_time=0.0,
                success=True,
                error_message="Skipped (file exists)"
            )
        
        try:
            start_time = time.time()
            
            # 调用噪声添加函数
            noisy_data, mask = noise.add_noise_avoid_coordinates(
                nifti_file=input_file,
                mni_coordinates=self.config.mni_coordinates,
                output_file=output_file,
                avoid_radius=avoid_radius,
                noise_type=self.config.noise_type,
                noise_params=noise_params,
                save_mask=self.config.save_mask,
                use_brain_mask=self.config.use_brain_mask
            )
            
            processing_time = time.time() - start_time
            
            # 计算统计信息（可选）
            original_stats = None
            noisy_stats = None
            mask_stats = None
            
            if self.config.save_intermediate:
                # 这里可以添加统计计算
                original_stats = {"mean": 0, "std": 0}  # 简化示例
                noisy_stats = {"mean": float(np.mean(noisy_data)), "std": float(np.std(noisy_data))}
                if mask is not None:
                    mask_stats = {"protected_voxels": int(np.sum(mask))}
            
            return ExperimentResult(
                experiment_id=experiment_id,
                input_file=input_file,
                output_file=output_file,
                avoid_radius=avoid_radius,
                noise_params=noise_params,
                processing_time=processing_time,
                success=True,
                original_stats=original_stats,
                noisy_stats=noisy_stats,
                mask_stats=mask_stats
            )
            
        except Exception as e:
            error_msg = f"处理失败: {str(e)}\n{traceback.format_exc()}"
            
            return ExperimentResult(
                experiment_id=experiment_id,
                input_file=input_file,
                output_file=output_file,
                avoid_radius=avoid_radius,
                noise_params=noise_params,
                processing_time=0.0,
                success=False,
                error_message=error_msg
            )
    
    def run_experiment(self) -> List[ExperimentResult]:
        """运行完整实验"""
        
        self.logger.info(f"开始实验: {self.experiment_name}")
        
        # 获取输入文件
        nii_dir = os.path.join(self.config.data_root, self.config.input_subdir)
        nii_files_all = directory_utils.get_nii_files_with_pattern(nii_dir)
        
        if self.config.max_subjects is not None:
            nii_files = nii_files_all[:self.config.max_subjects]
            # import pdb
            # pdb.set_trace()
        else:
            nii_files = nii_files_all
        
        self.logger.info(f"共处理 {len(nii_files)}/{len(nii_files_all)} 个NIfTI文件")
        
        # 生成参数组合
        param_combinations = self._generate_parameter_combinations()
        
        # 计算总任务数
        total_tasks = len(nii_files) * len(param_combinations)
        self.logger.info(f"总任务数: {total_tasks}")
        
        # 创建任务列表
        tasks = []
        for nii_file in nii_files:
            for avoid_radius, noise_params in param_combinations:
                tasks.append((nii_file, avoid_radius, noise_params))
        
        # 处理任务
        if self.config.max_workers > 1:
            self.results = self._run_parallel(tasks)
        else:
            self.results = self._run_sequential(tasks)
        
        # 保存结果
        self._save_results()
        
        # 生成报告
        self._generate_report()
        
        self.logger.info(f"实验完成: {self.experiment_name}")
        
        return self.results
    
    def _run_sequential(self, tasks: List[Tuple]) -> List[ExperimentResult]:
        """顺序处理"""
        
        results = []
        total_tasks = len(tasks)
        
        for i, (nii_file, avoid_radius, noise_params) in enumerate(tasks):
            self.logger.info(f"处理任务 {i+1}/{total_tasks}: {os.path.basename(nii_file)}")
            
            result = self._process_single_file(nii_file, avoid_radius, noise_params)
            results.append(result)
            
            if not result.success:
                self.logger.error(f"任务失败: {result.error_message}")
        
        return results
    
    def _run_parallel(self, tasks: List[Tuple]) -> List[ExperimentResult]:
        """并行处理"""
        
        results = []
        total_tasks = len(tasks)
        completed_tasks = 0
        
        with ProcessPoolExecutor(max_workers=self.config.max_workers) as executor:
            # 提交任务
            future_to_task = {
                executor.submit(self._process_single_file, nii_file, avoid_radius, noise_params): 
                (nii_file, avoid_radius, noise_params)
                for nii_file, avoid_radius, noise_params in tasks
            }
            
            # 收集结果
            for future in as_completed(future_to_task):
                nii_file, avoid_radius, noise_params = future_to_task[future]
                completed_tasks += 1
                
                try:
                    result = future.result()
                    results.append(result)
                    
                    self.logger.info(
                        f"完成 {completed_tasks}/{total_tasks}: "
                        f"{os.path.basename(nii_file)} (r={avoid_radius})"
                    )
                    
                    if not result.success:
                        self.logger.error(f"任务失败: {result.error_message}")
                        
                except Exception as e:
                    self.logger.error(f"任务异常: {nii_file}, {e}")
        
        return results
    
    def _save_results(self):
        """保存结果"""
        
        # 转换为DataFrame
        results_data = []
        for result in self.results:
            row = asdict(result)
            # 展开嵌套字典
            if row['noise_params']:
                for key, value in row['noise_params'].items():
                    row[f'noise_{key}'] = value
            if row['original_stats']:
                for key, value in row['original_stats'].items():
                    row[f'original_{key}'] = value
            if row['noisy_stats']:
                for key, value in row['noisy_stats'].items():
                    row[f'noisy_{key}'] = value
            if row['mask_stats']:
                for key, value in row['mask_stats'].items():
                    row[f'mask_{key}'] = value
            
            results_data.append(row)
        
        df = pd.DataFrame(results_data)
        
        # 保存CSV
        csv_file = os.path.join(self.experiment_dir, "results.csv")
        df.to_csv(csv_file, index=False, encoding='utf-8')
        
        # 保存JSON
        json_file = os.path.join(self.experiment_dir, "results.json")
        with open(json_file, 'w', encoding='utf-8') as f:
            json.dump([asdict(result) for result in self.results], f, indent=2, ensure_ascii=False)
        
        self.logger.info(f"结果已保存: {csv_file}, {json_file}")
    
    def _generate_report(self):
        """生成实验报告"""
        
        report_file = os.path.join(self.experiment_dir, "report.md")
        
        # 统计信息
        total_tasks = len(self.results)
        successful_tasks = sum(1 for r in self.results if r.success)
        failed_tasks = total_tasks - successful_tasks
        
        total_time = sum(r.processing_time for r in self.results if r.success)
        avg_time = total_time / successful_tasks if successful_tasks > 0 else 0
        
        # 参数统计
        radius_values = sorted(set(r.avoid_radius for r in self.results))
        noise_param_combinations = len(set(
            str(r.noise_params) for r in self.results
        ))
        
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write(f"# 噪声参数实验报告\n\n")
            f.write(f"**实验名称**: {self.experiment_name}\n")
            f.write(f"**实验时间**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            f.write(f"## 实验概况\n\n")
            f.write(f"- 总任务数: {total_tasks}\n")
            f.write(f"- 成功任务: {successful_tasks}\n")
            f.write(f"- 失败任务: {failed_tasks}\n")
            f.write(f"- 成功率: {successful_tasks/total_tasks*100:.1f}%\n")
            f.write(f"- 总处理时间: {total_time:.2f}秒\n")
            f.write(f"- 平均处理时间: {avg_time:.2f}秒/任务\n\n")
            
            f.write(f"## 参数设置\n\n")
            f.write(f"- 避开半径: {radius_values}\n")
            f.write(f"- 噪声参数组合数: {noise_param_combinations}\n")
            f.write(f"- MNI坐标数: {len(self.config.mni_coordinates)}\n\n")
            
            if failed_tasks > 0:
                f.write(f"## 失败任务\n\n")
                for result in self.results:
                    if not result.success:
                        f.write(f"- {os.path.basename(result.input_file)}: {result.error_message}\n")
        
        self.logger.info(f"报告已生成: {report_file}")

def create_default_config(data_root: str) -> ExperimentConfig:
    """创建默认实验配置"""
    
    return ExperimentConfig(
        data_root=data_root,
        input_subdir="nii_data_2507",
        output_base_subdir="nii_data_2507_noised",
        
        mni_coordinates=[
            [40, -20, 50],    # 右侧运动皮层
            [-40, -20, 50],   # 左侧运动皮层
            [0, 50, 20],      # 前额叶
        ],
        
        # 参数范围
        avoid_radius_list=[3, 6, 9, 12],
        noise_params_list=[
            None,  # 默认参数
            {'mean': 0, 'std': 25},
            {'mean': 0, 'std': 50},
            {'mean': 0, 'std': 75},
            {'mean': 0, 'std': 100},
        ],
        
        # 处理选项
        noise_type='gaussian',
        save_mask=False,
        use_brain_mask=True,
        
        # 实验选项
        max_workers=4,
        skip_existing=True,
        save_intermediate=True
    )

def run_noise_parameter_experiment():
    """运行噪声参数实验的主函数"""
    
    # 配置
    data_root = "/mnt/c/Works/ws/shoufa2025/data"
    config = create_default_config(data_root)
    
    # 运行实验
    runner = NoiseExperimentRunner(config, "noise_params_comparison")
    results = runner.run_experiment()
    
    # 简单分析
    print(f"\n实验完成! 共处理 {len(results)} 个任务")
    successful = [r for r in results if r.success]
    print(f"成功: {len(successful)}, 失败: {len(results) - len(successful)}")
    
    if successful:
        avg_time = np.mean([r.processing_time for r in successful])
        print(f"平均处理时间: {avg_time:.2f}秒")

def run_custom_experiment():
    """运行自定义实验示例"""
    
    data_root = "/mnt/c/Works/ws/shoufa2025/data"
    
    # 自定义配置
    config = ExperimentConfig(
        data_root=data_root,
        input_subdir="nii_data_2507",
        output_base_subdir="nii_data_custom_noise",
        
        mni_coordinates=[
            [0, 0, 0],        # 脑中心
            [20, 20, 20],     # 自定义点1
            [-20, -20, 20],   # 自定义点2
        ],
        
        # 精细化参数测试
        avoid_radius_list=[1, 2, 4, 6, 8, 10],
        noise_params_list=[
            {'mean': 0, 'std': 10},
            {'mean': 0, 'std': 30},
            {'mean': 0, 'std': 50},
            {'mean': 5, 'std': 25},   # 有偏移的噪声
            {'mean': -5, 'std': 25},
        ],
        
        max_workers=2,  # 减少并行度
        skip_existing=False  # 重新处理所有文件
    )
    
    # 运行
    runner = NoiseExperimentRunner(config, "custom_noise_experiment")
    results = runner.run_experiment()
    
    return results

if __name__ == "__main__":
    print(sys.path)
    # run_noise_parameter_experiment()
    
    # 或运行自定义实验
    # run_custom_experiment()
