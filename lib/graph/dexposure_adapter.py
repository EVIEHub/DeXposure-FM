"""
DeXposure 数据适配器 - 将 DeXposure 数据转换为 GraphPFN 格式

用途：
1. 将 DeXposure 的时序图数据转换为 GraphPFN 的 GraphData 格式
2. 支持时序编码
3. 支持时间感知的数据集划分
"""

import math
import numpy as np
import torch
from typing import Dict, Tuple
from sklearn.model_selection import train_test_split


class DeXposureToGraphPFNAdapter:
    """
    将 DeXposure 数据适配为 GraphData 格式

    用途：
    - 复用 DeXposureTemporalLoader 加载数据
    - 转换为 GraphPFN 标准的 GraphData 格式
    - 添加时序编码到节点特征
    - 创建 train/val/test 掩码
    """

    def __init__(
        self,
        data_path: str,      # DeXposure JSON 文件路径
        meta_path: str,      # 元数据 CSV 路径
        time_idx: int = 0,   # 时间索引
        split_strategy: str = 'temporal',  # 'temporal' 或 'random'
        label_idx: int = 0   # 使用哪个回归标签 (0=变化率, 1=绝对损失, 2=受影响程度)
    ):
        """
        初始化适配器

        Args:
            data_path: DeXposure JSON 文件路径
            meta_path: 元数据 CSV 路径
            time_idx: 要转换的时间快照索引
            split_strategy: 数据集划分策略
                - 'temporal': 时间感知划分（推荐）
                - 'random': 随机划分
            label_idx: 回归标签索引
                - 0: TVL 变化率（推荐）
                - 1: 绝对损失
                - 2: 受影响程度
        """
        from lib.graph.dexposure_temporal import DeXposureTemporalLoader

        self.loader = DeXposureTemporalLoader(
            data_path=data_path,
            meta_path=meta_path
        )
        self.time_idx = time_idx
        self.split_strategy = split_strategy
        self.label_idx = label_idx

        # 标签描述 (与 dexposure_temporal.py 的 _compute_labels 对应)
        label_names = ['log_change (推荐)', '相对变化率', '绝对损失', '受影响程度']
        
        print(f"✓ 初始化 DeXposure 适配器")
        print(f"  时间索引: {time_idx} / {len(self.loader.dates)}")
        print(f"  划分策略: {split_strategy}")
        print(f"  回归标签: {label_names[label_idx] if label_idx < len(label_names) else 'unknown'}")

    def convert(self) -> Dict:
        """
        转换指定时间快照为 GraphData 格式

        Returns:
            GraphData 字典，包含：
                - name: 数据集名称
                - graph: dgl.DGLGraph
                - labels: np.ndarray (num_nodes,)
                - masks: dict with 'train', 'val', 'test'
                - num_features: np.ndarray (num_nodes, feat_dim)
                - cat_features: None
                - ratio_features: None
        """
        print(f"\n转换时间快照 {self.time_idx}...")

        # 1. 加载 t 和 t+1 时刻的图
        print(f"  1. 加载时序图对...")
        graph_t, graph_t1, labels_all, _ = self.loader.get_temporal_pair(self.time_idx)

        # 2. 选择单一回归目标
        print(f"  2. 提取回归标签...")
        labels = labels_all[:, self.label_idx]  # 选择指定的标签维度

        # 3. 添加时序编码到节点特征
        print(f"  3. 添加时序编码...")
        features = self._add_temporal_encoding(
            graph_t.ndata['feat'],
            self.time_idx,
            len(self.loader.dates)
        )
        print(f"     原始特征维度: {graph_t.ndata['feat'].shape[1]}")
        print(f"     增强后维度: {features.shape[1]}")

        # 4. 创建 train/val/test 掩码
        print(f"  4. 创建数据集划分掩码...")
        masks = self._create_masks(
            num_nodes=graph_t.num_nodes(),
            split_strategy=self.split_strategy,
            time_idx=self.time_idx,
            num_total_times=len(self.loader.dates)
        )

        # 5. 构造 GraphData
        print(f"  5. 构造 GraphData...")
        graph_data = {
            'name': f'dexposure-week-{self.time_idx:02d}',
            'graph': graph_t,
            'labels': labels.numpy(),
            'masks': masks,
            'num_features': features.numpy(),
            'cat_features': None,
            'ratio_features': None
        }

        print(f"\n✓ 转换完成!")
        print(f"  数据集名称: {graph_data['name']}")
        print(f"  节点数: {graph_t.num_nodes()}")
        print(f"  边数: {graph_t.num_edges()}")
        print(f"  特征维度: {features.shape[1]}")
        print(f"  训练集: {masks['train'].sum()} | 验证集: {masks['val'].sum()} | 测试集: {masks['test'].sum()}")

        return graph_data

    def _add_temporal_encoding(
        self,
        features: torch.Tensor,
        time_idx: int,
        num_total_times: int
    ) -> torch.Tensor:
        """
        添加时序编码到节点特征

        方法：
        1. 归一化时间: time_idx / (num_total_times - 1)
        2. 位置编码: sin/cos 函数（4维）

        Args:
            features: 原始节点特征 (num_nodes, feat_dim)
            time_idx: 当前时间索引
            num_total_times: 总时间点数

        Returns:
            增强后的节点特征 (num_nodes, feat_dim + 5)
        """
        num_nodes = features.shape[0]

        # 方法 1: 归一化时间
        normalized_time = time_idx / max(num_total_times - 1, 1)

        # 方法 2: 位置编码 (4维)
        pos_enc_dim = 4
        pos_enc = torch.zeros(pos_enc_dim)
        for i in range(pos_enc_dim // 2):
            pos_enc[2*i] = math.sin(
                time_idx / (10000 ** (2*i / pos_enc_dim))
            )
            pos_enc[2*i+1] = math.cos(
                time_idx / (10000 ** (2*i / pos_enc_dim))
            )

        # 拼接时序特征
        temporal_features = torch.cat([
            pos_enc,
            torch.tensor([normalized_time])
        ])

        # 扩展到所有节点
        temporal_features = temporal_features.expand(num_nodes, -1)

        # 拼接到原始特征
        return torch.cat([features, temporal_features], dim=1)

    def _create_masks(
        self,
        num_nodes: int,
        split_strategy: str,
        time_idx: int,
        num_total_times: int
    ) -> Dict[str, np.ndarray]:
        """
        创建数据集划分掩码

        Args:
            num_nodes: 节点数量
            split_strategy: 'temporal' 或 'random'
            time_idx: 当前时间索引
            num_total_times: 总时间点数

        Returns:
            包含 'train', 'val', 'test' 的字典
        """
        if split_strategy == 'random':
            return self._random_split(num_nodes)
        elif split_strategy == 'temporal':
            return self._temporal_split(time_idx, num_total_times, num_nodes)
        else:
            raise ValueError(f"未知的划分策略: {split_strategy}")

    def _random_split(self, num_nodes: int) -> Dict[str, np.ndarray]:
        """
        随机划分节点

        比例: 训练 70% | 验证 15% | 测试 15%
        """
        indices = np.arange(num_nodes)

        # 划分为 train_val 和 test
        train_val_idx, test_idx = train_test_split(
            indices, test_size=0.15, random_state=42
        )

        # 划分 train_val 为 train 和 val
        val_size = 0.15 / 0.85  # 在剩余 85% 中，val 占 15%/85%
        train_idx, val_idx = train_test_split(
            train_val_idx, test_size=val_size, random_state=42
        )

        masks = {
            'train': self._index_to_mask(num_nodes, train_idx),
            'val': self._index_to_mask(num_nodes, val_idx),
            'test': self._index_to_mask(num_nodes, test_idx)
        }

        return masks

    def _temporal_split(
        self,
        time_idx: int,
        num_total_times: int,
        num_nodes: int
    ) -> Dict[str, np.ndarray]:
        """
        时间感知的划分策略

        策略：
        - 早期时间点（前 70%）: 主要用于训练
        - 中期时间点（70%-85%）: 主要用于验证
        - 后期时间点（后 15%）: 主要用于测试

        这模拟了真实预测场景：用过去预测未来
        """
        time_ratio = time_idx / num_total_times

        if time_ratio < 0.7:
            # 早期时间点：主要训练集
            ratios = {'train': 0.8, 'val': 0.1, 'test': 0.1}
        elif time_ratio < 0.85:
            # 中期时间点：主要验证集
            ratios = {'train': 0.3, 'val': 0.6, 'test': 0.1}
        else:
            # 后期时间点：主要测试集
            ratios = {'train': 0.1, 'val': 0.2, 'test': 0.7}

        # 为每个节点按概率分配
        rng = np.random.RandomState(42)
        masks = {}
        for i, node in enumerate(range(num_nodes)):
            # 生成随机数决定节点属于哪个集合
            rand_val = rng.random()
            cumulative = 0.0

            for part in ['train', 'val', 'test']:
                cumulative += ratios[part]
                if rand_val <= cumulative:
                    if part not in masks:
                        masks[part] = np.zeros(num_nodes, dtype=bool)
                    masks[part][i] = True
                    break

        # 确保所有掩码都已创建
        for part in ['train', 'val', 'test']:
            if part not in masks:
                masks[part] = np.zeros(num_nodes, dtype=bool)

        return masks

    def _index_to_mask(
        self,
        num_nodes: int,
        indices: np.ndarray
    ) -> np.ndarray:
        """
        将索引数组转换为布尔掩码

        Args:
            num_nodes: 总节点数
            indices: 索引数组

        Returns:
            布尔掩码数组
        """
        mask = np.zeros(num_nodes, dtype=bool)
        mask[indices] = True
        return mask
