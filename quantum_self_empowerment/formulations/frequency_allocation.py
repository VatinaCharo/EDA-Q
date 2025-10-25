"""超导量子芯片频率分配问题建模

将超导量子芯片的频率分配问题转化为QUBO形式，用于QAOA求解。

Classes:
    FrequencyAllocationQUBO: 频率分配QUBO建模类

Author: [待填写]
Created: 2025-10-16
Last Modified: 2025-10-16
"""

from typing import Dict, List, Tuple, Any, Optional
import numpy as np
from numpy.typing import NDArray
import networkx as nx

from .base import GraphBasedFormulation


class FrequencyAllocationQUBO(GraphBasedFormulation):
    """超导量子芯片频率分配QUBO建模类

    将频率分配问题转化为二次无约束二进制优化(QUBO)问题。

    Parameters
    ----------
    topology : nx.Graph
        量子比特连接拓扑图
    freq_range : tuple of float
        频率范围 (f_min, f_max)，单位GHz
    num_levels : int, optional
        频率离散化层数，默认为4
    min_separation : float, optional
        相邻量子比特最小频率间隔（GHz），默认为0.2
    penalty_weight : float, optional
        约束违反惩罚权重，默认为10.0

    Attributes
    ----------
    topology : nx.Graph
        量子比特拓扑
    frequency_options : np.ndarray
        可用的离散频率值
    num_qubits : int
        量子比特数量
    min_separation : float
        最小频率间隔
    penalty_weight : float
        惩罚权重

    Examples
    --------
    >>> import networkx as nx
    >>> topology = nx.cycle_graph(4)
    >>> formulation = FrequencyAllocationQUBO(
    ...     topology=topology,
    ...     freq_range=(4.5, 6.0),
    ...     num_levels=4,
    ...     min_separation=0.2
    ... )
    >>> qubo = formulation.build_qubo()
    >>> print(qubo.shape)
    (4, 4)

    Notes
    -----
    问题编码方式：
    - 简化编码：每个量子比特用1个二进制变量表示，变量值决定从哪组频率中选择
    - 完整编码（可选）：每个量子比特用log2(num_levels)个变量表示具体频率索引

    物理约束：
    - 相邻量子比特频率差必须大于min_separation
    - 频率必须在指定范围内

    References
    ----------
    .. [1] 基于现有的GAMS模型和物理约束
    """

    def __init__(
        self,
        topology: nx.Graph,
        freq_range: Tuple[float, float],
        num_levels: int = 4,
        min_separation: float = 0.2,
        penalty_weight: float = 10.0
    ):
        # 参数验证
        if num_levels <= 1:
            raise ValueError(f"num_levels must be > 1, got {num_levels}")

        if freq_range[0] >= freq_range[1]:
            raise ValueError(
                f"Invalid freq_range: {freq_range}. "
                f"Must have f_min < f_max"
            )

        if min_separation <= 0:
            raise ValueError(f"min_separation must be positive, got {min_separation}")

        if penalty_weight <= 0:
            raise ValueError(f"penalty_weight must be positive, got {penalty_weight}")

        # 初始化基类
        problem_params = {
            'freq_range': freq_range,
            'num_levels': num_levels,
            'min_separation': min_separation,
            'penalty_weight': penalty_weight
        }
        super().__init__(topology, problem_params)

        # 设置属性
        self.topology = topology
        self.num_qubits = len(topology.nodes())
        self.num_levels = num_levels
        self.min_separation = min_separation
        self.penalty_weight = penalty_weight

        # 生成离散频率选项
        self.frequency_options = self._generate_frequency_options(freq_range, num_levels)

        # QUBO变量数等于量子比特数（简化编码）
        self.num_variables = self.num_qubits

    def _generate_frequency_options(
        self,
        freq_range: Tuple[float, float],
        num_levels: int
    ) -> NDArray[np.float64]:
        """生成离散频率选项

        Parameters
        ----------
        freq_range : tuple
            频率范围
        num_levels : int
            离散化层数

        Returns
        -------
        NDArray[np.float64]
            频率选项数组
        """
        f_min, f_max = freq_range
        freq_step = (f_max - f_min) / (num_levels - 1)
        frequencies = np.array([f_min + i * freq_step for i in range(num_levels)])
        return frequencies

    def build_qubo(self) -> NDArray[np.float64]:
        """构建QUBO矩阵

        Returns
        -------
        NDArray[np.float64]
            QUBO矩阵 (n_qubits x n_qubits)

        Notes
        -----
        目标函数：最小化频率冲突

        编码方案（简化版）：
        - 每个量子比特i用1个二进制变量x_i表示
        - x_i = 0: 选择频率组A（偶数索引频率）
        - x_i = 1: 选择频率组B（奇数索引频率）
        - 实际频率根据量子比特ID和x_i值映射

        约束编码：
        - 相邻量子比特应选择不同的频率组（通过惩罚相同状态）
        """
        Q = np.zeros((self.num_qubits, self.num_qubits))

        # 为相邻量子比特添加惩罚：鼓励选择不同的比特值
        for i, j in self.topology.edges():
            # 添加惩罚项：如果x_i = x_j = 1，则惩罚
            Q[i, j] += self.penalty_weight
            Q[j, i] += self.penalty_weight

        # 添加偏置项：鼓励多样性
        for i in range(self.num_qubits):
            Q[i, i] -= 1.0

        self.qubo_matrix = Q
        return Q

    def decode_solution(self, bitstring: str) -> Dict[str, Any]:
        """解码量子解为频率分配方案

        Parameters
        ----------
        bitstring : str
            二进制解字符串

        Returns
        -------
        dict
            频率分配方案，包含:
            - 'frequencies': 字典 {qubit_id: frequency}
            - 'bitstring': 原始比特串
            - 'is_valid': 是否满足约束
        """
        # 清理比特串（去除空格）
        clean_bits = bitstring.replace(' ', '')

        # 如果比特串长度与num_qubits不匹配，尝试截断或填充
        if len(clean_bits) > self.num_qubits:
            # 比特串太长，取前num_qubits位
            import warnings
            warnings.warn(
                f"Bitstring length {len(clean_bits)} is longer than num_qubits {self.num_qubits}. "
                f"Truncating to first {self.num_qubits} bits. "
                f"Original: {clean_bits}, Using: {clean_bits[:self.num_qubits]}"
            )
            clean_bits = clean_bits[:self.num_qubits]
        elif len(clean_bits) < self.num_qubits:
            # 比特串太短，用0填充
            import warnings
            warnings.warn(
                f"Bitstring length {len(clean_bits)} is shorter than num_qubits {self.num_qubits}. "
                f"Padding with zeros."
            )
            clean_bits = clean_bits.zfill(self.num_qubits)

        # 解码为频率分配
        frequency_assignment = {}

        for i, bit in enumerate(clean_bits):
            bit_val = int(bit)

            # 简化映射策略：根据比特值和量子比特ID选择频率
            if bit_val == 0:
                # 选择组A的频率（根据量子比特ID）
                freq_idx = i % (self.num_levels // 2)
            else:
                # 选择组B的频率
                freq_idx = (i % (self.num_levels // 2)) + (self.num_levels // 2)

            freq_idx = min(freq_idx, self.num_levels - 1)  # 确保不越界
            frequency_assignment[i] = float(self.frequency_options[freq_idx])

        # 验证解的有效性
        is_valid = self.validate_solution({'frequencies': frequency_assignment})

        return {
            'frequencies': frequency_assignment,
            'bitstring': clean_bits,
            'is_valid': is_valid,
            'num_violations': self._count_violations(frequency_assignment)
        }

    def validate_solution(self, solution: Dict[str, Any]) -> bool:
        """验证频率分配方案的有效性

        Parameters
        ----------
        solution : dict
            包含'frequencies'字典的解

        Returns
        -------
        bool
            是否满足所有约束
        """
        frequencies = solution.get('frequencies', {})

        if len(frequencies) != self.num_qubits:
            return False

        # 检查频率范围
        f_min, f_max = self.problem_params['freq_range']
        for freq in frequencies.values():
            if freq < f_min or freq > f_max:
                return False

        # 检查相邻量子比特频率间隔
        for i, j in self.topology.edges():
            freq_diff = abs(frequencies[i] - frequencies[j])
            if freq_diff < self.min_separation:
                return False

        return True

    def _count_violations(self, frequencies: Dict[int, float]) -> int:
        """统计约束违反数量

        Parameters
        ----------
        frequencies : dict
            频率分配字典

        Returns
        -------
        int
            违反约束的数量
        """
        violations = 0

        for i, j in self.topology.edges():
            freq_diff = abs(frequencies[i] - frequencies[j])
            if freq_diff < self.min_separation:
                violations += 1

        return violations

    def evaluate_solution(self, solution: Dict[str, Any]) -> float:
        """评估解的质量

        Parameters
        ----------
        solution : dict
            频率分配方案

        Returns
        -------
        float
            评估得分（越小越好）
        """
        frequencies = solution.get('frequencies', {})

        # 计算总违反数
        violations = self._count_violations(frequencies)

        # 计算频率多样性（越高越好）
        unique_freqs = len(set(frequencies.values()))
        diversity_score = unique_freqs / self.num_qubits

        # 综合得分
        score = violations * 100 - diversity_score * 10

        return float(score)

    def get_frequency_statistics(
        self,
        solution: Dict[str, Any]
    ) -> Dict[str, Any]:
        """获取频率分配的统计信息

        Parameters
        ----------
        solution : dict
            频率分配方案

        Returns
        -------
        dict
            统计信息
        """
        frequencies = solution.get('frequencies', {})
        freq_values = list(frequencies.values())

        # 计算频率间隔
        freq_diffs = []
        for i, j in self.topology.edges():
            freq_diffs.append(abs(frequencies[i] - frequencies[j]))

        return {
            'mean_frequency': np.mean(freq_values),
            'std_frequency': np.std(freq_values),
            'num_unique_frequencies': len(set(freq_values)),
            'min_freq_difference': np.min(freq_diffs) if freq_diffs else 0.0,
            'mean_freq_difference': np.mean(freq_diffs) if freq_diffs else 0.0,
            'constraint_satisfaction_rate': 1.0 - (self._count_violations(frequencies) / len(self.topology.edges()))
        }

    def visualize_solution(
        self,
        solution: Dict[str, Any],
        ax=None
    ):
        """可视化频率分配方案

        Parameters
        ----------
        solution : dict
            频率分配方案
        ax : matplotlib.axes.Axes, optional
            绘图坐标轴

        Returns
        -------
        matplotlib.axes.Axes
            绘图坐标轴

        Notes
        -----
        需要matplotlib库
        """
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            raise ImportError("matplotlib is required for visualization")

        if ax is None:
            fig, ax = plt.subplots(figsize=(10, 8))

        frequencies = solution.get('frequencies', {})

        # 使用spring layout绘制拓扑
        pos = nx.spring_layout(self.topology)

        # 节点颜色基于频率
        node_colors = [frequencies.get(node, 0) for node in self.topology.nodes()]

        # 绘制节点
        nx.draw_networkx_nodes(
            self.topology,
            pos,
            node_color=node_colors,
            cmap='viridis',
            node_size=800,
            ax=ax
        )

        # 绘制边（根据是否违反约束着色）
        edge_colors = []
        for i, j in self.topology.edges():
            freq_diff = abs(frequencies[i] - frequencies[j])
            if freq_diff < self.min_separation:
                edge_colors.append('red')  # 违反约束
            else:
                edge_colors.append('gray')  # 满足约束

        nx.draw_networkx_edges(
            self.topology,
            pos,
            edge_color=edge_colors,
            width=2,
            ax=ax
        )

        # 添加标签（显示频率）
        labels = {
            node: f"Q{node}\n{frequencies.get(node, 0):.2f}"
            for node in self.topology.nodes()
        }
        nx.draw_networkx_labels(
            self.topology,
            pos,
            labels,
            font_size=10,
            font_weight='bold',
            ax=ax
        )

        ax.set_title('频率分配方案可视化\n(红边=违反约束, 灰边=满足约束)', fontsize=14)
        ax.axis('off')

        return ax


if __name__ == "__main__":
    # 简单测试
    import networkx as nx

    print("FrequencyAllocationQUBO模块加载成功!")

    # 创建测试拓扑
    topology = nx.cycle_graph(4)

    # 创建问题实例
    formulation = FrequencyAllocationQUBO(
        topology=topology,
        freq_range=(4.5, 6.0),
        num_levels=4,
        min_separation=0.2
    )

    print(f"量子比特数: {formulation.num_qubits}")
    print(f"频率选项: {formulation.frequency_options}")

    # 构建QUBO
    Q = formulation.build_qubo()
    print(f"QUBO矩阵形状: {Q.shape}")
    print(f"QUBO矩阵:\n{Q}")

    # 测试解码
    test_solution = formulation.decode_solution("1010")
    print(f"\n测试解: {test_solution}")
    print(f"解有效性: {test_solution['is_valid']}")
