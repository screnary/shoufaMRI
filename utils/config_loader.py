# -*- coding: utf-8 -*-
"""
配置文件读取和验证模块
支持新的 mni_coordinates 格式
"""

import os, sys
import yaml
import logging
from typing import Dict, List, Optional, Union, Any, Tuple
from dataclasses import dataclass, field, asdict
from pathlib import Path
import jsonschema
from jsonschema import validate, ValidationError
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
    __package__ = 'shoufaMRI.utils'

from ..core.exp_frame_noise import ExperimentConfig  # 避免循环导入

cur_path = os.path.abspath(__file__)  #/mnt/c/Works/ws/shoufa2025/code/shoufaMRI/utils/config_loader.py

class CompactDumper(yaml.SafeDumper):
    """自定义YAML输出器，让3维坐标更紧凑"""
    def represent_list(self, data):
        # 如果是3个数字的列表（坐标），使用flow style
        if (len(data) == 3 and 
            all(isinstance(x, (int, float)) for x in data)):
            return self.represent_sequence('tag:yaml.org,2002:seq', data, flow_style=True)
        # 其他列表使用默认格式
        return self.represent_sequence('tag:yaml.org,2002:seq', data, flow_style=False)

CompactDumper.add_representer(list, CompactDumper.represent_list)

@dataclass
class MNICoordinateGroup:
    """MNI坐标组数据类 - 支持多个坐标点"""
    name: str
    coordinates: List[List[float]]  # 坐标点列表
    description: str = ""
    
    def get_coordinate_list(self) -> List[List[float]]:
        """获取坐标列表格式 (兼容原始格式)"""
        return [list(coord) for coord in self.coordinates]
    
    def get_coordinate_count(self) -> int:
        """获取坐标点数量"""
        return len(self.coordinates)

@dataclass
class NoiseParamConfig:
    """噪声参数配置数据类"""
    name: str
    params: Optional[Dict[str, Union[int, float]]]
    description: str = ""

@dataclass
class ExperimentConfigYAML:
    """基于YAML的实验配置数据类"""
    
    # 实验信息
    experiment_name: str = ""
    experiment_description: str = ""
    experiment_author: str = ""
    experiment_version: str = "1.0"
    
    # 数据路径
    data_root: str = ""
    input_subdir: str = ""
    output_base_subdir: str = ""
    
    # MNI坐标组
    mni_coordinate_groups: List[MNICoordinateGroup] = field(default_factory=list)
    
    # 参数设置
    avoid_radius_list: List[float] = field(default_factory=list)
    noise_params_list: List[NoiseParamConfig] = field(default_factory=list)
    
    # 处理选项
    noise_type: str = "gaussian"
    save_mask: bool = False
    use_brain_mask: bool = True
    
    # 执行选项
    max_workers: int = 4
    skip_existing: bool = True
    save_intermediate: bool = True
    max_subjects: Optional[int] = None
    
    # 输出选项
    save_config: bool = True
    save_csv: bool = True
    save_json: bool = True
    generate_report: bool = True
    
    # 日志配置
    log_level: str = "INFO"
    console_output: bool = True
    file_output: bool = True
    
    # 高级选项
    memory_limit_gb: int = 8
    max_retries: int = 3
    retry_delay: int = 5
    monitor_memory: bool = True
    monitor_cpu: bool = True
    backup_original: bool = False
    
    # 扩展
    custom_validators: List[str] = field(default_factory=list)
    post_processors: List[str] = field(default_factory=list)
    plugins: List[str] = field(default_factory=list)
    
    def get_default_coordinate_group(self) -> Optional[MNICoordinateGroup]:
        """获取默认坐标组 (第一个或名为'whole'的组)"""
        if not self.mni_coordinate_groups:
            return None
        
        # 优先查找名为'whole'的组
        for group in self.mni_coordinate_groups:
            if group.name.lower() == 'whole':
                return group
        
        # 如果没找到，返回第一个
        return self.mni_coordinate_groups[0]
    
    def get_coordinate_group_by_name(self, name: str) -> Optional[MNICoordinateGroup]:
        """根据名称获取坐标组"""
        for group in self.mni_coordinate_groups:
            if group.name == name:
                return group
        return None


class ConfigLoader:
    """配置文件加载器"""
    
    def __init__(self, logger: Optional[logging.Logger] = None):
        self.logger = logger or logging.getLogger(__name__)
        
    def create_default_config_file(self, filepath: str) -> str:
        """创建默认配置文件"""
        
        default_config = self._get_default_config_dict()
        
        # 确保目录存在
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        # 修改这部分 - 添加自定义Dumper
        with open(filepath, 'w', encoding='utf-8') as f:
            yaml.dump(default_config, f, 
                    Dumper=CompactDumper,  # 使用自定义dumper
                    default_flow_style=False, 
                    allow_unicode=True, 
                    indent=2, 
                    sort_keys=False)
        
        self.logger.info(f"默认配置文件已创建: {filepath}")
        return filepath
    
    def _get_default_config_dict(self) -> Dict[str, Any]:
        """获取默认配置字典"""
        
        return {
            'experiment': {
                'name': '',
                'description': '噪声参数对比实验',
                'author': 'researcher',
                'version': '1.0'
            },
            'data': {
                'root': '/mnt/c/Works/ws/shoufa2025/data',
                'input_subdir': 'nii_data_2507',
                'output_base_subdir': 'nii_data_noised'
            },
            'mni_coordinates': [
                {
                    'name': 'whole',
                    'description': '避开DMN, SN, CEN网络节点的完整坐标集',
                    'coordinates': [
                        [-18, 24, 53],
                        [22, 26,  51],
                        [-18, -1, 65],
                        [20, 4,   64],
                        [-27, 43, 31],
                        [30, 37,  36],
                        [-42, 13, 36],
                        [42, 11,  39],
                        [-28, 56, 12],
                        [28, 55,  17],
                        [-41, 41, 16],
                        [42, 44,  14],
                        [-33, 23, 45],
                        [42, 27,  39],
                        [-32, 4,  55],
                        [34, 8,   54],
                        [-26, 60, -6],
                        [25, 61,  -4],
                        [-65, -30, -12],
                        [65, -29,  -13],
                        [-53, 2,  -30],
                        [51, 6,  -32],
                        [-59, -58, 4],
                        [60, -53, 3],
                        [-58, -20, -9],
                        [58, -16, -10],
                        [-27, -7, -34],
                        [28, -8, -33],
                        [-25, -25, -26],
                        [26, -23, -27],
                        [-28, -32, -18],
                        [30, -30, -18],
                        [-19, -12, -30],
                        [19, -10, -30],
                        [-23, 2, -32],
                        [22, 1, -36],
                        [-17, -39, -10],
                        [19, -36, -11],
                        [-16, -60, 63],
                        [19, -57, 65],
                        [-27, -59, 54],
                        [31, -54, 53],
                        [-34, -80, 29],
                        [45, -71, 20],
                        [-38, -61, 46],
                        [39, -65, 44],
                        [-51, -33, 42],
                        [47, -35, 45],
                        [-56, -49, 38],
                        [57, -44, 38],
                        [-47, -65, 26],
                        [53, -54, 25],
                        [-53, -31, 23],
                        [55, -26, 26],
                        [-5, -63, 51],
                        [6, -65, 51],
                        [-8, -47, 57],
                        [7, -47, 58],
                        [-12, -67, 25],
                        [16, -64, 25],
                        [-6, -55, 34],
                        [6, -54, 35],
                        [-36, -20, 10],
                        [37, -18, 8],
                        [-32, 14, -13],
                        [33, 14, -13],
                        [-34, 18, 1],
                        [36, 18, 1],
                        [-38, -4, -9],
                        [39, -2, -9],
                        [-38, -8, 8],
                        [39, -7, 8],
                        [-38, 5, 5],
                        [38, 5, 5],
                        [-4, -39, 31],
                        [4, -37, 32],
                        [-3, 8, 25],
                        [5, 22, 12],
                        [-6, 34, 21],
                        [5, 28, 27],
                        [-8, -47, 10],
                        [9, -44, 11],
                        [-5, 7, 37],
                        [4, 6, 38],
                        [-7, -23, 41],
                        [6, -20, 40],
                        [-4, 39, -2],
                        [5, 41, 6],
                        [-19, -2, -20],
                        [19, -2, -19],
                        [-27, -4, -20],
                        [28, -3, -20],
                        [-22, -14, -19],
                        [22, -12, -20],
                        [-28, -30, -10],
                        [29, -27, -10],
                        [-7, -12, 5],
                        [7, -11, 6],
                        [-18, -13, 3],
                        [12, -14, 1],
                        [-18, -23, 4],
                        [18, -22, 3],
                        [-7, -14, 7],
                        [3, -13, 5],
                        [-16, -24, 6],
                        [15, -25, 6],
                        [-15, -28, 4],
                        [13, -27, 8],
                        [-12, -22, 13],
                        [10, -14, 14],
                        [-11, -14, 2],
                        [13, -16, 7]
                    ]
                }
            ],
            'parameters': {
                'avoid_radius': {
                    'values': [3, 6, 9, 12],
                    'description': '保护区域半径，单位毫米'
                },
                'noise_params': [
                    {
                        'name': 'default',
                        'params': None,
                        'description': '使用默认噪声参数'
                    },
                    {
                        'name': 'low_noise',
                        'params': {'mean': 0, 'std': 25},
                        'description': '低强度高斯噪声'
                    },
                    {
                        'name': 'medium_noise', 
                        'params': {'mean': 0, 'std': 50},
                        'description': '中等强度高斯噪声'
                    },
                    {
                        'name': 'high_noise',
                        'params': {'mean': 0, 'std': 75},
                        'description': '高强度高斯噪声'
                    }
                ]
            },
            'processing': {
                'noise_type': 'gaussian',
                'save_mask': False,
                'use_brain_mask': True
            },
            'execution': {
                'max_workers': 4,
                'skip_existing': True,
                'save_intermediate': True
            },
            'output': {
                'save_config': True,
                'save_csv': True,
                'save_json': True,
                'generate_report': True
            },
            'logging': {
                'level': 'INFO',
                'console_output': True,
                'file_output': True
            },
            'advanced': {
                'memory_limit_gb': 8,
                'max_retries': 3,
                'retry_delay': 5,
                'monitor_memory': True,
                'monitor_cpu': True,
                'backup_original': False
            },
            'extensions': {
                'custom_validators': [],
                'post_processors': [],
                'plugins': []
            }
        }
    
    def load_config(self, config_path: str) -> ExperimentConfigYAML:
        """从YAML文件加载配置"""
        
        if not os.path.exists(config_path):
            self.logger.warning(f"配置文件不存在: {config_path}")
            self.logger.info("将创建默认配置文件")
            self.create_default_config_file(config_path)
        
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                config_dict = yaml.safe_load(f)
            
            self.logger.info(f"成功加载配置文件: {config_path}")
            
            # 验证配置
            self._validate_config(config_dict)
            
            # 转换为数据类
            return self._dict_to_config(config_dict)
            
        except yaml.YAMLError as e:
            self.logger.error(f"YAML解析错误: {e}")
            raise
        except Exception as e:
            self.logger.error(f"配置加载失败: {e}")
            raise
    
    def _validate_config(self, config_dict: Dict[str, Any]):
        """验证配置文件"""
        
        # 定义配置schema
        schema = {
            "type": "object",
            "required": ["data", "mni_coordinates", "parameters"],
            "properties": {
                "data": {
                    "type": "object",
                    "required": ["root", "input_subdir", "output_base_subdir"],
                    "properties": {
                        "root": {"type": "string"},
                        "input_subdir": {"type": "string"},
                        "output_base_subdir": {"type": "string"}
                    }
                },
                "mni_coordinates": {
                    "type": "array",
                    "minItems": 1,
                    "items": {
                        "type": "object",
                        "required": ["name", "coordinates"],
                        "properties": {
                            "name": {"type": "string"},
                            "coordinates": {
                                "type": "array",
                                "minItems": 1,
                                "items": {
                                    "type": "array",
                                    "items": {"type": "number"},
                                    "minItems": 3,
                                    "maxItems": 3
                                }
                            },
                            "description": {"type": "string"}
                        }
                    }
                },
                "parameters": {
                    "type": "object",
                    "required": ["avoid_radius", "noise_params"],
                    "properties": {
                        "avoid_radius": {
                            "type": "object",
                            "required": ["values"],
                            "properties": {
                                "values": {
                                    "type": "array",
                                    "items": {"type": "number", "minimum": 0}
                                }
                            }
                        },
                        "noise_params": {
                            "type": "array",
                            "minItems": 1,
                            "items": {
                                "type": "object",
                                "required": ["name"],
                                "properties": {
                                    "name": {"type": "string"},
                                    "params": {
                                        "oneOf": [
                                            {"type": "null"},
                                            {
                                                "type": "object",
                                                "properties": {
                                                    "mean": {"type": "number"},
                                                    "std": {"type": "number", "minimum": 0}
                                                }
                                            }
                                        ]
                                    },
                                    "description": {"type": "string"}
                                }
                            }
                        }
                    }
                }
            }
        }
        
        try:
            validate(instance=config_dict, schema=schema)
            self.logger.info("配置文件验证通过")
        except ValidationError as e:
            self.logger.error(f"配置文件验证失败: {e.message}")
            raise
    
    def _dict_to_config(self, config_dict: Dict[str, Any]) -> ExperimentConfigYAML:
        """将字典转换为配置数据类"""
        
        # 提取实验信息
        exp_info = config_dict.get('experiment', {})
        
        # 提取数据路径
        data_info = config_dict['data']
        
        # 提取MNI坐标组
        mni_coordinate_groups = []
        for coord_group_info in config_dict['mni_coordinates']:
            # 转换坐标格式
            coordinates = [tuple(coord) for coord in coord_group_info['coordinates']]
            
            mni_coordinate_groups.append(MNICoordinateGroup(
                name=coord_group_info['name'],
                coordinates=coordinates,
                description=coord_group_info.get('description', '')
            ))
        
        # 提取参数
        params_info = config_dict['parameters']
        avoid_radius_list = params_info['avoid_radius']['values']
        
        noise_params_list = []
        for noise_info in params_info['noise_params']:
            noise_params_list.append(NoiseParamConfig(
                name=noise_info['name'],
                params=noise_info.get('params'),
                description=noise_info.get('description', '')
            ))
        
        # 提取其他配置
        processing = config_dict.get('processing', {})
        execution = config_dict.get('execution', {})
        output = config_dict.get('output', {})
        logging_cfg = config_dict.get('logging', {})
        advanced = config_dict.get('advanced', {})
        extensions = config_dict.get('extensions', {})
        
        return ExperimentConfigYAML(
            # 实验信息
            experiment_name=exp_info.get('name', ''),
            experiment_description=exp_info.get('description', ''),
            experiment_author=exp_info.get('author', ''),
            experiment_version=exp_info.get('version', '1.0'),
            
            # 数据路径
            data_root=data_info['root'],
            input_subdir=data_info['input_subdir'],
            output_base_subdir=data_info['output_base_subdir'],
            
            # 坐标和参数
            mni_coordinate_groups=mni_coordinate_groups,
            avoid_radius_list=avoid_radius_list,
            noise_params_list=noise_params_list,
            
            # 处理选项
            noise_type=processing.get('noise_type', 'gaussian'),
            save_mask=processing.get('save_mask', False),
            use_brain_mask=processing.get('use_brain_mask', True),
            
            # 执行选项
            max_workers=execution.get('max_workers', 4),
            skip_existing=execution.get('skip_existing', True),
            save_intermediate=execution.get('save_intermediate', True),
            max_subjects=execution.get('max_subjects', None),
            
            # 输出选项
            save_config=output.get('save_config', True),
            save_csv=output.get('save_csv', True),
            save_json=output.get('save_json', True),
            generate_report=output.get('generate_report', True),
            
            # 日志配置
            log_level=logging_cfg.get('level', 'INFO'),
            console_output=logging_cfg.get('console_output', True),
            file_output=logging_cfg.get('file_output', True),
            
            # 高级选项
            memory_limit_gb=advanced.get('memory_limit_gb', 8),
            max_retries=advanced.get('max_retries', 3),
            retry_delay=advanced.get('retry_delay', 5),
            monitor_memory=advanced.get('monitor_memory', True),
            monitor_cpu=advanced.get('monitor_cpu', True),
            backup_original=advanced.get('backup_original', False),
            
            # 扩展
            custom_validators=extensions.get('custom_validators', []),
            post_processors=extensions.get('post_processors', []),
            plugins=extensions.get('plugins', [])
        )
    
    def convert_to_original_config(self, yaml_config: ExperimentConfigYAML, 
                                 coordinate_group_name: str = "whole") -> 'ExperimentConfig':
        """将YAML配置转换为原始ExperimentConfig"""
        
        # 获取指定的坐标组
        coord_group = yaml_config.get_coordinate_group_by_name(coordinate_group_name)
        if coord_group is None:
            coord_group = yaml_config.get_default_coordinate_group()
        
        if coord_group is None:
            raise ValueError("没有找到可用的MNI坐标组")
        
        # 提取MNI坐标
        mni_coordinates = coord_group.coordinates
        
        # 提取噪声参数
        noise_params_list = [param.params for param in yaml_config.noise_params_list]
        
        return ExperimentConfig(
            data_root=yaml_config.data_root,
            input_subdir=yaml_config.input_subdir,
            output_base_subdir=yaml_config.output_base_subdir,
            mni_coordinates=mni_coordinates,
            avoid_radius_list=yaml_config.avoid_radius_list,
            noise_params_list=noise_params_list,
            noise_type=yaml_config.noise_type,
            save_mask=yaml_config.save_mask,
            use_brain_mask=yaml_config.use_brain_mask,
            max_workers=yaml_config.max_workers,
            max_subjects=yaml_config.max_subjects,
            skip_existing=yaml_config.skip_existing,
            save_intermediate=yaml_config.save_intermediate
        )

def load_experiment_config(config_path: str = "config.yaml", 
                         coordinate_group: str = "whole") -> 'ExperimentConfig':
    """加载实验配置的便捷函数"""
    
    loader = ConfigLoader()
    yaml_config = loader.load_config(config_path)
    return loader.convert_to_original_config(yaml_config, coordinate_group)

def create_sample_configs():
    """创建示例配置文件"""
    config_root = os.path.join(os.path.dirname(os.path.dirname(cur_path)), "config")
    config_fpath = os.path.join(config_root, "config_default.yaml")
    
    loader = ConfigLoader()

    # 1. 默认配置
    quick_config = loader._get_default_config_dict()
    quick_config['parameters']['avoid_radius']['values'] = [6]
    quick_config['parameters']['noise_params'] = [
        {'name': 'default', 'params': None, 'description': '默认参数'},
        {'name': 'test', 'params': {'mean': 0, 'std': 50}, 'description': '测试参数'}
    ]
    quick_config['execution']['max_workers'] = 1
    
    with open(config_fpath, 'w', encoding='utf-8') as f:
        yaml.dump(quick_config, f, 
                 Dumper=CompactDumper,  # 添加这行
                 default_flow_style=False, 
                 allow_unicode=True, indent=2, sort_keys=False)
    
    # 3. 多坐标组配置
    multi_coord_config = loader._get_default_config_dict()
    
    # 添加额外的坐标组
    multi_coord_config['mni_coordinates'].extend([
        {
            'name': 'motor_only',
            'description': '仅运动皮层相关区域',
            'coordinates': [
                [-32, 4, 55],   # 左侧运动皮层
                [34, 8, 54],    # 右侧运动皮层
                [-33, 23, 45],  # 左侧中央前回
                [42, 27, 39]    # 右侧中央前回
            ]
        },
        {
            'name': 'frontal_only',
            'description': '仅前额叶相关区域',
            'coordinates': [
                [-27, 43, 31],  # 左侧前额叶
                [30, 37, 36],   # 右侧前额叶
                [-26, 60, -6],  # 左侧眶额皮层
                [25, 61, -4]    # 右侧眶额皮层
            ]
        }
    ])
    
    with open(os.path.join(config_root,"config_multi_coords_test.yaml"), 'w', encoding='utf-8') as f:
        yaml.dump(multi_coord_config, f, 
                 Dumper=CompactDumper,  # 添加这行
                 default_flow_style=False,
                 allow_unicode=True, indent=2, sort_keys=False)
    
    print("示例配置文件已创建:")
    print("- config_default.yaml: 默认配置 (112个坐标点)")
    print("- config_multi_coords_test.yaml: 附加多组坐标点配置")


def print_coordinate_info(config_path: str = "config.yaml"):
    """打印坐标信息"""
    
    loader = ConfigLoader()
    yaml_config = loader.load_config(config_path)
    
    print(f"\n=== MNI坐标组信息 ===")
    
    for i, group in enumerate(yaml_config.mni_coordinate_groups):
        print(f"\n{i+1}. 坐标组: {group.name}")
        print(f"   描述: {group.description}")
        print(f"   坐标点数: {group.get_coordinate_count()}")
        print(f"   坐标列表:")
        
        for j, coord in enumerate(group.coordinates):
            print(f"     {j+1:2d}. ({coord[0]:6.1f}, {coord[1]:6.1f}, {coord[2]:6.1f})")
        
        if i == 0:  # 只显示第一组的详细信息
            print(f"   坐标范围:")
            coords_array = list(group.coordinates)
            x_coords = [c[0] for c in coords_array]
            y_coords = [c[1] for c in coords_array]
            z_coords = [c[2] for c in coords_array]
            
            print(f"     X: [{min(x_coords):6.1f}, {max(x_coords):6.1f}]")
            print(f"     Y: [{min(y_coords):6.1f}, {max(y_coords):6.1f}]")
            print(f"     Z: [{min(z_coords):6.1f}, {max(z_coords):6.1f}]")

if __name__ == "__main__":
    # 创建示例配置文件
    create_sample_configs()
    
    # 打印坐标信息
    print_coordinate_info("../config/config_default.yaml")
    
    # 测试加载不同坐标组
    try:
        print(f"\n=== 测试配置加载 ===")
        
        # 加载默认坐标组 (whole)
        config_whole = load_experiment_config("../config/config_default.yaml", "whole")
        print(f"✓ 成功加载 'whole' 坐标组")
        print(f"  坐标点数: {len(config_whole.mni_coordinates)}")
        
        # 如果存在多坐标组配置，测试其他组
        if os.path.exists("../config/config_multi_coords.yaml"):
            config_motor = load_experiment_config("../config/config_multi_coords.yaml", "motor_only") 
            print(f"✓ 成功加载 'motor_only' 坐标组")
            print(f"  坐标点数: {len(config_motor.mni_coordinates)}")
            
    except Exception as e:
        print(f"✗ 配置加载失败: {e}")
