# src/causal_discovery/wrappers/pc.py
# 作者: LiuZijian(liuzj109@163.com & liuzijian-cs@shu.edu.cn)
# 时间: 2025-09-19 16:30
# 描述: PC算法实现 (PC Algorithm Implementation)

from typing import Union, Dict, Tuple, Optional, List

import numpy as np
import pandas as pd
from causallearn.search.ConstraintBased.PC import pc
from causallearn.utils.PCUtils.BackgroundKnowledge import BackgroundKnowledge
from causallearn.graph.GraphNode import GraphNode

from .base_model import BaseModel


class PC(BaseModel):
    """
    PC算法实现类

    基于causal-learn库的PC算法封装，提供与GlobalState集成的接口
    """

    def __init__(self, state, params: Optional[Dict] = None):
        # 默认参数配置
        default_params = {
            "alpha": 0.05,  # 独立性检验的显著性水平
            "indep_test": "fisherz",  # 独立性检验方法
            "stable": True,  # 是否使用稳定版本
            "uc_rule": 0,  # 无向边冲突解决规则
            "uc_priority": -1,  # 无向边优先级
            "mvpc": False,  # 是否处理缺失值
            "correction_name": "MV_Crtn_Fisher_Z",  # 多重检验校正
            "background_knowledge": None,  # 背景知识
            "verbose": False,  # 详细输出
            "show_progress": True,  # 显示进度
        }

        # 合并默认参数和用户参数
        merged_params = {**default_params, **(params or {})}

        # 调用基类构造函数
        super().__init__(state, merged_params)

        # 从GlobalState同步相关参数
        self._sync_with_global_state()

    @property
    def name(self) -> str:
        """返回算法名称"""
        return "PC"

    def _sync_with_global_state(self):
        """与GlobalState同步参数"""
        # 如果GlobalState中有alpha设置，优先使用
        if self.state.statistics.alpha is not None:
            self._params["alpha"] = self.state.statistics.alpha

        # 检查是否需要处理缺失值
        if self.state.statistics.miss_ratio:
            total_missing = sum(
                item.get("missing_ratio", 0)
                for item in self.state.statistics.miss_ratio
            )
            if total_missing > 0:
                self._params["mvpc"] = True
                if self._params.get("verbose"):
                    print(
                        f"Detected missing values (total ratio: {total_missing:.2%}), enabling MVPC"
                    )

        # 整合先验知识（如果有）
        if self.state.results.prior_knowledge is not None:
            self._params["background_knowledge"] = self.state.results.prior_knowledge

    def fit(self, data: Union[pd.DataFrame, np.ndarray]) -> Tuple[np.ndarray, Dict]:
        """
        执行PC算法进行因果发现

        Args:
            data: 输入数据

        Returns:
            tuple: (邻接矩阵, 附加信息字典)
        """
        # 数据预处理
        data, node_names = self._preprocess_data(data)

        # 处理高相关特征（如果启用）
        if self.state.algorithm.handle_correlated_features:
            data, node_names = self._handle_correlated_features(data, node_names)

        # 准备PC算法参数
        pc_params = self._prepare_pc_params(node_names)

        # 运行PC算法
        try:
            cg = pc(
                data.values if isinstance(data, pd.DataFrame) else data, **pc_params
            )
        except Exception as e:
            if self._params.get("verbose"):
                print(f"PC algorithm failed: {e}")
            raise

        # 转换结果
        adj_matrix = self._convert_graph_to_adjacency(cg.G.graph)

        # 收集信息
        info = self._collect_algorithm_info(cg, node_names)

        # 更新GlobalState的结果
        self._update_global_state(adj_matrix, info)

        return adj_matrix, info

    def _preprocess_data(
        self, data: Union[pd.DataFrame, np.ndarray]
    ) -> Tuple[Union[pd.DataFrame, np.ndarray], List[str]]:
        """数据预处理"""
        if isinstance(data, pd.DataFrame):
            # 移除域索引列
            if "domain_index" in data.columns:
                data = data.drop(columns=["domain_index"])

            # 移除用户指定的特征
            for drop_list in self.state.user_data.drop_features.values():
                data = data.drop(
                    columns=[col for col in drop_list if col in data.columns]
                )

            node_names = list(data.columns)
        else:
            node_names = [f"X{i}" for i in range(data.shape[1])]

        return data, node_names

    def _handle_correlated_features(
        self, data: Union[pd.DataFrame, np.ndarray], node_names: List[str]
    ) -> Tuple[Union[pd.DataFrame, np.ndarray], List[str]]:
        """处理高相关特征"""
        if not isinstance(data, pd.DataFrame):
            data = pd.DataFrame(data, columns=node_names)

        threshold = self.state.algorithm.correlation_threshold
        corr_matrix = data.corr().abs()

        # 找出高相关特征对
        high_corr_pairs = []
        for i in range(len(corr_matrix.columns)):
            for j in range(i + 1, len(corr_matrix.columns)):
                corr_value = corr_matrix.iloc[i, j]
                if isinstance(corr_value, (int, float)) and corr_value > threshold:
                    high_corr_pairs.append(
                        (corr_matrix.columns[i], corr_matrix.columns[j])
                    )

        if high_corr_pairs and self._params.get("verbose"):
            print(
                f"Found {len(high_corr_pairs)} highly correlated feature pairs (threshold: {threshold})"
            )

        # 这里可以添加处理逻辑，比如移除或合并高相关特征
        # 目前仅记录到state中
        if high_corr_pairs:
            self.state.user_data.high_corr_feature_groups = high_corr_pairs

        return data, list(data.columns)

    def _prepare_pc_params(self, node_names: List[str]) -> Dict:
        """准备PC算法参数"""
        pc_params = self._params.copy()

        # 处理背景知识
        if pc_params.get("background_knowledge") is not None:
            pc_params["background_knowledge"] = self._create_background_knowledge(
                pc_params["background_knowledge"], node_names
            )

        # 添加节点名称
        pc_params["node_names"] = node_names

        return pc_params

    def _create_background_knowledge(
        self, bk_spec: Union[Dict, BackgroundKnowledge], node_names: List[str]
    ) -> BackgroundKnowledge:
        """创建背景知识对象"""
        if isinstance(bk_spec, BackgroundKnowledge):
            return bk_spec

        nodes = [GraphNode(name) for name in node_names]
        node_dict = {name: node for name, node in zip(node_names, nodes)}

        bk = BackgroundKnowledge()

        if isinstance(bk_spec, dict):
            # 处理禁止边
            if "forbidden_edges" in bk_spec:
                for edge in bk_spec.get("forbidden_edges", []):
                    if len(edge) == 2 and all(v in node_dict for v in edge):
                        bk.add_forbidden_by_node(node_dict[edge[0]], node_dict[edge[1]])

            # 处理必须边
            if "required_edges" in bk_spec:
                for edge in bk_spec.get("required_edges", []):
                    if len(edge) == 2 and all(v in node_dict for v in edge):
                        bk.add_required_by_node(node_dict[edge[0]], node_dict[edge[1]])

            # 处理时间层级
            if "tiers" in bk_spec:
                for var_name, tier_level in bk_spec.get("tiers", {}).items():
                    if var_name in node_dict:
                        bk.add_node_to_tier(node_dict[var_name], tier_level)

        return bk

    def _convert_graph_to_adjacency(self, graph_matrix: np.ndarray) -> np.ndarray:
        """
        转换图格式为标准邻接矩阵

        标准格式: 0=无边, 1=有向边(i->j), 2=无向边, 3=双向边
        """
        n = graph_matrix.shape[0]
        adj_matrix = np.zeros((n, n), dtype=int)

        for i in range(n):
            for j in range(i + 1, n):
                edge_i_j = graph_matrix[i, j]
                edge_j_i = graph_matrix[j, i]

                if edge_i_j == -1 and edge_j_i == 1:
                    adj_matrix[j, i] = 1  # i -> j
                elif edge_i_j == 1 and edge_j_i == -1:
                    adj_matrix[i, j] = 1  # j -> i
                elif edge_i_j == -1 and edge_j_i == -1:
                    adj_matrix[i, j] = adj_matrix[j, i] = 2  # i -- j
                elif edge_i_j == 1 and edge_j_i == 1:
                    adj_matrix[i, j] = adj_matrix[j, i] = 3  # i <-> j

        return adj_matrix

    def _collect_algorithm_info(self, cg, node_names: List[str]) -> Dict:
        """收集算法运行信息"""
        info = {
            "algorithm": self.name,
            "parameters": self.get_params(),
            "node_names": node_names,
            "causal_graph": cg,
        }

        # 添加可用的额外信息
        optional_attrs = [
            "sepset",
            "test_statistic",
            "p_values",
            "definite_UC",
            "definite_non_UC",
        ]

        for attr in optional_attrs:
            if hasattr(cg, attr):
                info[attr] = getattr(cg, attr)

        return info

    def _update_global_state(self, adj_matrix: np.ndarray, info: Dict):
        """更新GlobalState的结果"""
        # 存储原始结果
        self.state.results.raw_result = adj_matrix
        self.state.results.raw_info = info

        # 转换为边列表格式
        edges = self._matrix_to_edges(adj_matrix, info["node_names"])
        self.state.results.raw_edges = edges

        # 如果有分离集信息，可以用于后续分析
        if "sepset" in info:
            self.state.results.raw_info["sepset"] = info["sepset"]

    def _matrix_to_edges(self, adj_matrix: np.ndarray, node_names: List[str]) -> Dict:
        """将邻接矩阵转换为边列表"""
        edges = {
            "directed": [],  # 有向边
            "undirected": [],  # 无向边
            "bidirected": [],  # 双向边
        }

        n = adj_matrix.shape[0]
        for i in range(n):
            for j in range(n):
                if adj_matrix[i, j] == 1:
                    edges["directed"].append((node_names[j], node_names[i]))
                elif adj_matrix[i, j] == 2 and i < j:  # 避免重复
                    edges["undirected"].append((node_names[i], node_names[j]))
                elif adj_matrix[i, j] == 3 and i < j:  # 避免重复
                    edges["bidirected"].append((node_names[i], node_names[j]))

        return edges
    
