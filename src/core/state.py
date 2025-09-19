# src/core/state.py
# 作者: LiuZijian(liuzj109@163.com & liuzijian-cs@shu.edu.cn)
# 时间: 2025-09-18 20:40
# 引用: https://github.com/Lancelot39/Causal-Copilot
# 描述: 全局状态管理 (Global State Management)

from typing import List, Dict, Optional, Any

import pandas as pd
import numpy as np
from pydantic import BaseModel, Field


class UserData(BaseModel):
    """用户数据状态 (User Data State)"""
    model_config = {"arbitrary_types_allowed": True}
    
    data_name: Optional[str] = None
    raw_data: Optional[pd.DataFrame] = None
    processed_data: Optional[pd.DataFrame] = None
    ground_truth: Optional[np.ndarray] = None
    knowledge_docs: Optional[str] = None
    knowledge_docs_for_user: Optional[str] = None
    output_dir: Optional[str] = None
    selected_features: Optional[object] = None
    important_features: Optional[object] = None
    high_corr_feature_groups: Optional[object] = None
    visual_selected_features: Optional[object] = None
    drop_features: Dict[str, List[str]] = Field(default_factory=lambda: {
        "user": [],
        "llm": [],
        "high_corr": []
    })
    accept_cpdag: bool = True
    
    
class Statistics(BaseModel):
    """统计信息 (Statistical Information)"""
    # 数据特征 (Data Characteristics)
    sample_size: Optional[int] = None
    feature_number: Optional[int] = None
    miss_ratio: List[Dict] = Field(default_factory=list)
    sparsity: Optional[float] = None
    data_type: Optional[str] = None
    data_type_column: Optional[str] = None
    
    # 假设条件 (Assumptions)
    # linearity: Optional[bool] = None
    # gaussian_error: Optional[bool] = None
    heterogeneous: Optional[bool] = None
    # domain_index: Optional[str] = None

    # Text：
    description: Optional[str] = None
    
    # 测试参数 (Test Parameters)
    alpha: float = 0.1
    boot_num: int = 20
    num_test: int = 100
    ratio: float = 0.5
    

class Algorithm(BaseModel):
    """算法配置 (Algorithm Configuration)"""
    selected_algorithm: Optional[str] = None
    selected_reason: Optional[str] = None
    algorithm_candidates: Dict[str, Any] = Field(default_factory=dict)
    algorithm_optimum: Optional[str] = None
    algorithm_arguments: Dict[str, Any] = Field(default_factory=dict)
    algorithm_arguments_json: Optional[object] = None
    waiting_minutes: float = 1440.0
    handle_correlated_features: Optional[bool] = True # 自动处理高相关特征，后续加入
    correlation_threshold: float = 0.99
    gpu_available: bool = False


class Results(BaseModel):
    """结果状态 (Results State)"""
    model_config = {"arbitrary_types_allowed": True}
    
    raw_result: Optional[object] = None
    raw_pos: Optional[object] = None
    raw_edges: Optional[Dict] = None
    raw_info: Optional[Dict] = None
    converted_graph: Optional[str] = None
    lagged_graph: Optional[object] = None
    metrics: Optional[Dict] = None
    revised_graph: Optional[np.ndarray] = None
    revised_edges: Optional[Dict] = None
    revised_metrics: Optional[Dict] = None
    bootstrap_probability: Optional[np.ndarray] = None
    bootstrap_check_dict: Optional[Dict] = None
    llm_errors: Optional[Dict] = None
    bootstrap_errors: List[Dict] = Field(default_factory=list)
    eda_result: Optional[Dict] = None
    prior_knowledge: Optional[object] = None
    refutation_analysis: Optional[object] = None
    report_selected_index: Optional[object] = None

# background_knowledge = {
#     "forbidden_edges": [["A", "B"], ["C", "D"]],
#     "required_edges": [["E", "F"]],
#     "tiers": {"A": 1, "B": 2}
# }

class Inference(BaseModel):
    """推理状态 (Inference State)"""
    hte_algo_json: Optional[Dict] = None
    hte_model_y_json: Optional[Dict] = None
    hte_model_T_json: Optional[Dict] = None
    hte_model_param: Optional[Dict] = None
    cycle_detection_result: Optional[Dict] = Field(default_factory=dict)  # 🔹 Stores detected cycles
    editing_history: List[Dict] = Field(default_factory=list)  # 🔹 Tracks cycle resolution steps
    inference_result: Optional[Dict] = Field(default_factory=dict)  # 🔹 Stores final inference output
    task_index: Optional[int] = -1
    task_info: Optional[Dict] = None


class GlobalState(BaseModel):
    """全局状态管理器 (Global State Manager)"""
    user_data: UserData = Field(default_factory=UserData)
    statistics: Statistics = Field(default_factory=Statistics)
    algorithm: Algorithm = Field(default_factory=Algorithm)
    inference: Inference = Field(default_factory=Inference)
    results: Results = Field(default_factory=Results)
