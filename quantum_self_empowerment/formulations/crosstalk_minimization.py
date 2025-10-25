"""超导量子芯片串扰最小化布局优化

本模块实现将量子比特布局优化问题转化为QUBO形式。

问题描述
--------
给定N个量子比特及其工作频率，需要在M×M的2D网格上安排它们的物理位置，
使得串扰最小化。串扰主要来源于：
1. 频率接近的量子比特物理距离太近
2. 有连接的量子比特距离太远（增加布线串扰）

Classes
-------
CrosstalkMinimizationQUBO : QUBO建模类

Author: [待填写]
Created: 2025-10-21
"""

from typing import Dict, Any, List, Tuple
import numpy as np
from numpy.typing import NDArray
import networkx as nx
import itertools


class CrosstalkMinimizationQUBO:
    """串扰最小化布局优化QUBO建模

    将量子比特布局问题编码为QUBO矩阵。

    Parameters
    ----------
    num_qubits : int
        量子比特数量
    grid_size : int
        布局网格大小（grid_size × grid_size）
    frequencies : List[float]
        每个量子比特的工作频率（GHz）
    topology : nx.Graph, optional
        量子比特连接拓扑图，默认为None（无连接）
    freq_threshold : float, optional
        频率接近阈值（GHz），默认0.1
    penalty_weight : float, optional
        约束违反惩罚权重，默认100.0
    objective_weights : Dict[str, float], optional
        目标权重字典

    Attributes
    ----------
    num_variables : int
        QUBO变量数量（num_qubits × grid_size × grid_size）
    var_to_position : Dict[int, Tuple[int, int, int]]
        变量索引到(qubit_id, x, y)的映射

    Examples
    --------
    >>> import networkx as nx
    >>> topology = nx.cycle_graph(4)
    >>> frequencies = [5.0, 5.1, 5.5, 5.6]
    >>> formulation = CrosstalkMinimizationQUBO(
    ...     num_qubits=4,
    ...     grid_size=3,
    ...     frequencies=frequencies,
    ...     topology=topology
    ... )
    >>> Q = formulation.build_qubo()
    """

    def __init__(
        self,
        num_qubits: int,
        grid_size: int,
        frequencies: List[float],
        topology: nx.Graph = None,
        freq_threshold: float = 0.1,
        penalty_weight: float = 100.0,
        objective_weights: Dict[str, float] = None
    ):
        self.num_qubits = num_qubits
        self.grid_size = grid_size
        self.frequencies = np.array(frequencies)
        self.topology = topology if topology is not None else nx.Graph()
        self.freq_threshold = freq_threshold
        self.penalty_weight = penalty_weight

        # 默认目标权重
        default_weights = {
            'freq_separation': 1.0,      # 频率分离：频率接近的要远离
            'topology_proximity': 1.0,   # 拓扑接近：有连接的要靠近
            'compactness': 0.3          # 紧凑性：整体布局紧凑
        }
        self.objective_weights = objective_weights if objective_weights else default_weights

        # 构建变量映射
        self._build_variable_mapping()

    def _build_variable_mapping(self):
        """构建变量索引与位置的映射"""
        self.var_to_position = {}
        self.position_to_var = {}

        var_idx = 0
        for qubit_id in range(self.num_qubits):
            for x in range(self.grid_size):
                for y in range(self.grid_size):
                    self.var_to_position[var_idx] = (qubit_id, x, y)
                    self.position_to_var[(qubit_id, x, y)] = var_idx
                    var_idx += 1

        self.num_variables = var_idx

    def build_qubo(self) -> NDArray[np.float64]:
        """构建QUBO矩阵

        Returns
        -------
        NDArray[np.float64]
            QUBO矩阵 (n_vars × n_vars)
        """
        n_vars = self.num_variables
        Q = np.zeros((n_vars, n_vars))

        # 1. 约束：每个量子比特恰好在一个位置
        self._add_one_position_constraint(Q)

        # 2. 约束：每个位置最多一个量子比特
        self._add_one_qubit_constraint(Q)

        # 3. 目标：频率接近的量子比特要远离
        self._add_frequency_separation_objective(Q)

        # 4. 目标：有连接的量子比特要靠近
        self._add_topology_proximity_objective(Q)

        # 5. 目标：布局紧凑性
        self._add_compactness_objective(Q)

        return Q

    def _add_one_position_constraint(self, Q: NDArray[np.float64]):
        """添加约束：每个量子比特恰好在一个位置

        对于每个量子比特q，Σ_{x,y} x_{q,x,y} = 1
        惩罚形式：penalty * (Σ_{x,y} x_{q,x,y} - 1)^2
        """
        penalty = self.penalty_weight

        for qubit_id in range(self.num_qubits):
            # 找到该量子比特的所有位置变量
            position_vars = []
            for x in range(self.grid_size):
                for y in range(self.grid_size):
                    var_idx = self.position_to_var[(qubit_id, x, y)]
                    position_vars.append(var_idx)

            # 展开 (Σx_i - 1)^2 = Σx_i^2 - 2Σx_i + 1
            # 对角线项：x_i^2 = x_i（二进制）
            for var_idx in position_vars:
                Q[var_idx, var_idx] += penalty * (1 - 2)

            # 交叉项：2 * x_i * x_j
            for v1, v2 in itertools.combinations(position_vars, 2):
                Q[v1, v2] += penalty
                Q[v2, v1] += penalty

    def _add_one_qubit_constraint(self, Q: NDArray[np.float64]):
        """添加约束：每个位置最多一个量子比特

        对于每个位置(x,y)，Σ_q x_{q,x,y} <= 1
        惩罚形式：penalty * Σ_{q1<q2} x_{q1,x,y} * x_{q2,x,y}
        """
        penalty = self.penalty_weight

        for x in range(self.grid_size):
            for y in range(self.grid_size):
                # 找到该位置的所有量子比特变量
                qubit_vars = []
                for qubit_id in range(self.num_qubits):
                    var_idx = self.position_to_var[(qubit_id, x, y)]
                    qubit_vars.append(var_idx)

                # 添加交叉惩罚项
                for v1, v2 in itertools.combinations(qubit_vars, 2):
                    Q[v1, v2] += penalty
                    Q[v2, v1] += penalty

    def _add_frequency_separation_objective(self, Q: NDArray[np.float64]):
        """添加目标：频率接近的量子比特要远离

        对于频率差 < freq_threshold 的量子比特对，
        惩罚它们的物理距离近。
        """
        weight = self.objective_weights['freq_separation']

        # 找出频率接近的量子比特对
        for q1 in range(self.num_qubits):
            for q2 in range(q1 + 1, self.num_qubits):
                freq_diff = abs(self.frequencies[q1] - self.frequencies[q2])

                if freq_diff < self.freq_threshold:
                    # 频率接近，需要物理远离
                    # 惩罚强度与频率接近程度成正比
                    proximity_penalty = weight * (1 - freq_diff / self.freq_threshold)

                    # 对所有位置组合，距离近的增加惩罚
                    for x1, y1 in itertools.product(range(self.grid_size), repeat=2):
                        for x2, y2 in itertools.product(range(self.grid_size), repeat=2):
                            distance = np.sqrt((x1 - x2)**2 + (y1 - y2)**2)

                            if distance > 0:  # 避免除零
                                # 距离近的惩罚更大（1/distance）
                                penalty_value = proximity_penalty / distance

                                var1 = self.position_to_var[(q1, x1, y1)]
                                var2 = self.position_to_var[(q2, x2, y2)]

                                Q[var1, var2] += penalty_value
                                Q[var2, var1] += penalty_value

    def _add_topology_proximity_objective(self, Q: NDArray[np.float64]):
        """添加目标：有连接的量子比特要靠近

        对于拓扑中有边连接的量子比特对，
        奖励它们的物理距离近。
        """
        if self.topology.number_of_edges() == 0:
            return

        weight = self.objective_weights['topology_proximity']

        for q1, q2 in self.topology.edges():
            # 有连接，需要物理靠近
            # 对所有位置组合，距离近的减少成本（奖励）
            for x1, y1 in itertools.product(range(self.grid_size), repeat=2):
                for x2, y2 in itertools.product(range(self.grid_size), repeat=2):
                    distance = np.sqrt((x1 - x2)**2 + (y1 - y2)**2)

                    # 距离作为成本（距离远惩罚大）
                    cost_value = weight * distance

                    var1 = self.position_to_var[(q1, x1, y1)]
                    var2 = self.position_to_var[(q2, x2, y2)]

                    Q[var1, var2] += cost_value
                    Q[var2, var1] += cost_value

    def _add_compactness_objective(self, Q: NDArray[np.float64]):
        """添加目标：布局紧凑性

        鼓励量子比特靠近网格中心，避免过于分散。
        """
        weight = self.objective_weights['compactness']
        center = self.grid_size / 2.0

        for qubit_id in range(self.num_qubits):
            for x in range(self.grid_size):
                for y in range(self.grid_size):
                    # 到中心的距离
                    distance_to_center = np.sqrt((x - center)**2 + (y - center)**2)

                    # 距离中心远的增加成本
                    cost = weight * distance_to_center

                    var_idx = self.position_to_var[(qubit_id, x, y)]
                    Q[var_idx, var_idx] += cost

    def decode_solution(self, bitstring: str) -> Dict[str, Any]:
        """解码二进制解为布局方案

        Parameters
        ----------
        bitstring : str
            二进制解字符串

        Returns
        -------
        dict
            布局方案，包含:
            - 'layout': Dict[int, Tuple[int, int]] 量子比特ID到位置的映射
            - 'grid': NDArray 2D网格可视化（qubit_id或-1表示空）
            - 'is_valid': bool 是否满足所有约束
            - 'metrics': dict 布局性能指标
        """
        # 清理比特串
        clean_bits = bitstring.replace(' ', '')

        # 长度校验
        if len(clean_bits) != self.num_variables:
            clean_bits = clean_bits[:self.num_variables].zfill(self.num_variables)

        # 解码布局
        layout = {}
        grid = np.full((self.grid_size, self.grid_size), -1, dtype=int)

        for var_idx, bit in enumerate(clean_bits):
            if bit == '1':
                qubit_id, x, y = self.var_to_position[var_idx]
                layout[qubit_id] = (x, y)
                grid[x, y] = qubit_id

        # 验证解的有效性
        is_valid = self._validate_layout(layout, grid)

        # 计算性能指标
        metrics = self._compute_layout_metrics(layout, grid)

        return {
            'layout': layout,
            'grid': grid,
            'is_valid': is_valid,
            'metrics': metrics,
            'bitstring': clean_bits
        }

    def _validate_layout(self, layout: Dict[int, Tuple[int, int]],
                        grid: NDArray[np.int64]) -> bool:
        """验证布局方案的有效性

        检查：
        1. 每个量子比特恰好在一个位置
        2. 每个位置最多一个量子比特
        """
        # 检查所有量子比特都有位置
        if len(layout) != self.num_qubits:
            return False

        # 检查位置唯一性
        positions = set(layout.values())
        if len(positions) != self.num_qubits:
            return False

        return True

    def _compute_layout_metrics(self, layout: Dict[int, Tuple[int, int]],
                                grid: NDArray[np.int64]) -> Dict[str, float]:
        """计算布局性能指标"""
        if len(layout) == 0:
            return {
                'avg_freq_pair_distance': 0.0,
                'avg_topology_distance': 0.0,
                'compactness_score': 0.0,
                'crosstalk_score': 0.0
            }

        # 1. 频率接近的量子比特对的平均距离
        freq_distances = []
        for q1 in range(self.num_qubits):
            for q2 in range(q1 + 1, self.num_qubits):
                freq_diff = abs(self.frequencies[q1] - self.frequencies[q2])
                if freq_diff < self.freq_threshold:
                    if q1 in layout and q2 in layout:
                        x1, y1 = layout[q1]
                        x2, y2 = layout[q2]
                        dist = np.sqrt((x1 - x2)**2 + (y1 - y2)**2)
                        freq_distances.append(dist)

        avg_freq_distance = np.mean(freq_distances) if freq_distances else 0.0

        # 2. 拓扑连接的量子比特对的平均距离
        topo_distances = []
        if self.topology.number_of_edges() > 0:
            for q1, q2 in self.topology.edges():
                if q1 in layout and q2 in layout:
                    x1, y1 = layout[q1]
                    x2, y2 = layout[q2]
                    dist = np.sqrt((x1 - x2)**2 + (y1 - y2)**2)
                    topo_distances.append(dist)

        avg_topo_distance = np.mean(topo_distances) if topo_distances else 0.0

        # 3. 紧凑性得分（到中心的平均距离）
        center = self.grid_size / 2.0
        center_distances = []
        for qubit_id, (x, y) in layout.items():
            dist = np.sqrt((x - center)**2 + (y - center)**2)
            center_distances.append(dist)

        compactness = np.mean(center_distances) if center_distances else 0.0

        # 4. 综合串扰得分（越小越好）
        # 串扰 = 频率接近的距离近 + 拓扑连接的距离远
        crosstalk = 0.0
        if freq_distances:
            crosstalk += sum(1.0 / (d + 0.1) for d in freq_distances)  # 距离近惩罚大
        if topo_distances:
            crosstalk += sum(d for d in topo_distances)  # 距离远惩罚大

        return {
            'avg_freq_pair_distance': avg_freq_distance,
            'avg_topology_distance': avg_topo_distance,
            'compactness_score': compactness,
            'crosstalk_score': crosstalk,
            'num_freq_pairs': len(freq_distances),
            'num_topo_edges': len(topo_distances)
        }

    def get_layout_summary(self, solution: Dict[str, Any]) -> str:
        """生成布局方案摘要"""
        layout = solution['layout']
        metrics = solution['metrics']
        is_valid = solution['is_valid']

        summary = "=" * 60 + "\n"
        summary += "布局设计方案摘要\n"
        summary += "=" * 60 + "\n\n"

        summary += f"有效性: {'✅ 有效' if is_valid else '❌ 无效'}\n\n"

        summary += "基本信息:\n"
        summary += f"  量子比特数: {self.num_qubits}\n"
        summary += f"  网格大小: {self.grid_size}×{self.grid_size}\n"
        summary += f"  已分配位置: {len(layout)}\n\n"

        summary += "性能指标:\n"
        summary += f"  频率接近对平均距离: {metrics['avg_freq_pair_distance']:.2f}\n"
        summary += f"  拓扑连接平均距离: {metrics['avg_topology_distance']:.2f}\n"
        summary += f"  紧凑性得分: {metrics['compactness_score']:.2f}\n"
        summary += f"  综合串扰得分: {metrics['crosstalk_score']:.2f}\n"
        summary += f"  频率接近对数: {metrics['num_freq_pairs']}\n"
        summary += f"  拓扑连接数: {metrics['num_topo_edges']}\n\n"

        summary += "=" * 60 + "\n"

        return summary
