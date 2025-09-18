# src/core/initializer.py
# 作者: LiuZijian(liuzj109@163.com & liuzijian-cs@shu.edu.cn)
# 时间: 2025-09-18 21:00
# 描述: 状态初始化器 (State Initializer)

import os
import logging
from pathlib import Path
from datetime import datetime

import torch
import pandas as pd

from ..llm import LLMClient
from .state import GlobalState

class Initializer(object):
    """状态初始化器 (State Initializer)"""
    
    def __init__(self, 
                 data_path: str,
                 accept_CPDAG: bool = True,
                 ):
        """
        初始化 (Initialize)
        
        Args:
            args: 命令行参数 (Command line arguments)
        """
        self.logger = logging.getLogger(__name__)
        if not logging.getLogger().hasHandlers():
            logging.basicConfig(
                level=logging.INFO,
                format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
                datefmt="%Y-%m-%d %H:%M:%S",
            )

        self.data_path = data_path
        self.state = GlobalState()
        self.client = self._load_llm_client()
        self._load_data()
        self._setup_output_paths()
        self.state.algorithm.gpu_available = torch.cuda.is_available()
        self.state.user_data.accept_cpdag = accept_CPDAG
    
    def _load_algorithms(self):
        """加载可用算法 (Load available algorithms)"""
        algorithms = [algo.split('.')[0] for algo in os.listdir('src/causal_discovery/context/algorithms') if algo.endswith('.txt') and 'tagging' not in algo and 'guideline' not in algo]  
        algorithms = ', '.join(algorithms)
        self.logger.debug(f"Available algorithms: {len(algorithms.split(', '))}")
        return algorithms
    
    def _load_data(self):
        """加载数据 (Load data)"""
        if not os.path.exists(self.data_path):
            raise FileNotFoundError(f"文件不存在: {self.data_path}")
        file_extension = os.path.splitext(self.data_path)[1].lower()
        try:
            if file_extension == '.csv':
                try:
                    df = pd.read_csv(self.data_path, encoding='utf-8')
                except UnicodeDecodeError:
                    df = pd.read_csv(self.data_path, encoding='gbk')
            elif file_extension in ['.xlsx', '.xls']:
                df = pd.read_excel(self.data_path)
            else:
                raise ValueError(f"不支持的文件格式: {file_extension}。仅支持 .csv, .xlsx, .xls")
            self.state.user_data.raw_data = df
            self.logger.info(f"数据加载成功: {self.data_path} (Data loaded successfully)")
        except Exception as e:
            raise Exception(f"读取文件时出错: {str(e)}")
        
    def _load_llm_client(self):
        """初始化LLM客户端 (Initialize LLM client)"""
        return LLMClient(
            provider="default",
            model=os.getenv("MODEL", "gpt-5"),
            api_key=os.getenv("API_KEY"),
            base_url=os.getenv("LLM_API_URL")
        )
    
    def _setup_output_paths(self):
        """设置输出路径 (Setup output paths)"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_base = Path(f'output/{self.data_path}/{timestamp}')
        self.state.user_data.output_dir = str(output_base)

if __name__ == "__main__":
    initializer = Initializer(data_path="demo/data/data.csv")
    print(f"Output directory: {initializer.state.user_data.output_dir}")