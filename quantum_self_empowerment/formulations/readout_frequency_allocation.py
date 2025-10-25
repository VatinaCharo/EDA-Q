"""超导量子芯片读出频率分配问题建模

将超导量子芯片的读出频率分配问题转化为QUBO形式。
与计算频率分配的区别在于约束类型和频率范围。

Classes:
    ReadoutFrequencyAllocationQUBO: 读出频率分配QUBO建模类
    JointFrequencyAllocationQUBO: 计算+读出频率联合优化建模类

Author: [待填写]
Created: 2025-10-16
"""

from typing import Dict, List, Tuple, Any, Optional
import numpy as np
from numpy.typing import NDArray
import networkx as nx
import warnings

from .base import GraphBasedFormulation


class ReadoutFrequencyAllocationQUBO(GraphBasedFormulation):
    """超导量子芯片读出频率分配QUBO建模类

    为每个量子比特的读出谐振器分配频率，需要考虑：
    1. 读出-读出串扰（相邻读出谐振器频率间隔）
    2. 读出-计算串扰（读出频率不能太接近计算频率）

    Parameters
    ----------
    topology : nx.Graph
        量子比特连接拓扑图
    readout_freq_range : tuple of float
        读出频率范围 (f_min, f_max)，单位GHz
    compute_frequencies : dict, optional
        已分配的计算频率 {qubit_id: frequency}
        如果提供，将考虑读出-计算串扰约束
    num_levels : int, optional
        频率离散化层数，默认为4
    min_readout_separation : float, optional
        相邻读出谐振器最小频率间隔（GHz），默认为0.15
    min_readout_compute_separation : float, optional
        读出频率与计算频率最小间隔（GHz），默认为0.5
    penalty_weight : float, optional
        约束违反惩罚权重，默认为10.0

    Examples
    --------
    >>> import networkx as nx
    >>> topology = nx.cycle_graph(4)
    >>> compute_freqs = {0: 5.0, 1: 5.2, 2: 5.4, 3: 5.6}
    >>> formulation = ReadoutFrequencyAllocationQUBO(
    ...     topology=topology,
    ...     readout_freq_range=(6.5, 7.5),
    ...     compute_frequencies=compute_freqs,
    ...     num_levels=4
    ... )
    >>> qubo = formulation.build_qubo()
    """

    def __init__(
        self,
        topology: nx.Graph,
        readout_freq_range: Tuple[float, float],
        compute_frequencies: Optional[Dict[int, float]] = None,
        num_levels: int = 4,
        min_readout_separation: float = 0.15,
        min_readout_compute_separation: float = 0.5,
        penalty_weight: float = 10.0
    ):
        # 参数验证
        if num_levels <= 1:
            raise ValueError(f"num_levels must be > 1, got {num_levels}")

        if readout_freq_range[0] >= readout_freq_range[1]:
            raise ValueError(
                f"Invalid readout_freq_range: {readout_freq_range}. "
                f"Must have f_min < f_max"
            )

        if min_readout_separation <= 0:
            raise ValueError(f"min_readout_separation must be positive")

        if min_readout_compute_separation <= 0:
            raise ValueError(f"min_readout_compute_separation must be positive")

        # 初始化基类
        problem_params = {
            'readout_freq_range': readout_freq_range,
            'compute_frequencies': compute_frequencies,
            'num_levels': num_levels,
            'min_readout_separation': min_readout_separation,
            'min_readout_compute_separation': min_readout_compute_separation,
            'penalty_weight': penalty_weight
        }
        super().__init__(topology, problem_params)

        # 设置属性
        self.topology = topology
        self.num_qubits = len(topology.nodes())
        self.num_levels = num_levels
        self.min_readout_separation = min_readout_separation
        self.min_readout_compute_separation = min_readout_compute_separation
        self.penalty_weight = penalty_weight
        self.compute_frequencies = compute_frequencies or {}

        # 生成离散读出频率选项
        self.readout_frequency_options = self._generate_frequency_options(
            readout_freq_range, num_levels
        )

        # QUBO变量数等于量子比特数（简化编码）
        self.num_variables = self.num_qubits

    def _generate_frequency_options(
        self,
        freq_range: Tuple[float, float],
        num_levels: int
    ) -> NDArray[np.float64]:
        """生成离散频率选项"""
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
        目标函数：
        1. 最小化相邻读出谐振器的频率冲突
        2. 如果提供了计算频率，避免读出-计算频率过近

        编码方案（简化版）：
        - 每个量子比特i用1个二进制变量x_i表示
        - x_i = 0: 选择读出频率组A
        - x_i = 1: 选择读出频率组B
        """
        Q = np.zeros((self.num_qubits, self.num_qubits))

        # 1. 相邻读出谐振器频率分离约束
        for i, j in self.topology.edges():
            # 添加惩罚项：如果x_i = x_j = 1，则惩罚
            Q[i, j] += self.penalty_weight
            Q[j, i] += self.penalty_weight

        # 2. 添加偏置项：鼓励多样性
        for i in range(self.num_qubits):
            Q[i, i] -= 1.0

        # 3. 如果提供了计算频率，添加读出-计算分离约束
        if self.compute_frequencies:
            # 对于每个量子比特，如果其计算频率已知
            # 我们倾向于选择远离计算频率的读出频率
            for qubit_id, compute_freq in self.compute_frequencies.items():
                if qubit_id < self.num_qubits:
                    # 计算每个读出频率选项与计算频率的距离
                    # 选择更远的选项
                    distances = np.abs(self.readout_frequency_options - compute_freq)

                    # 如果组B的平均距离更大，鼓励选择x=1
                    group_a_dist = np.mean(distances[:self.num_levels//2])
                    group_b_dist = np.mean(distances[self.num_levels//2:])

                    if group_b_dist > group_a_dist:
                        Q[qubit_id, qubit_id] -= 2.0  # 鼓励x=1
                    else:
                        Q[qubit_id, qubit_id] += 2.0  # 鼓励x=0

        self.qubo_matrix = Q
        return Q

    def decode_solution(self, bitstring: str) -> Dict[str, Any]:
        """解码量子解为读出频率分配方案

        Parameters
        ----------
        bitstring : str
            二进制解字符串

        Returns
        -------
        dict
            读出频率分配方案
        """
        # 清理比特串
        clean_bits = bitstring.replace(' ', '')

        # 长度匹配处理
        if len(clean_bits) > self.num_qubits:
            warnings.warn(
                f"Bitstring length {len(clean_bits)} > num_qubits {self.num_qubits}. "
                f"Truncating to first {self.num_qubits} bits."
            )
            clean_bits = clean_bits[:self.num_qubits]
        elif len(clean_bits) < self.num_qubits:
            warnings.warn(
                f"Bitstring length {len(clean_bits)} < num_qubits {self.num_qubits}. "
                f"Padding with zeros."
            )
            clean_bits = clean_bits.zfill(self.num_qubits)

        # 解码为读出频率分配
        readout_frequency_assignment = {}

        for i, bit in enumerate(clean_bits):
            bit_val = int(bit)

            # 根据比特值和量子比特ID选择读出频率
            if bit_val == 0:
                freq_idx = i % (self.num_levels // 2)
            else:
                freq_idx = (i % (self.num_levels // 2)) + (self.num_levels // 2)

            freq_idx = min(freq_idx, self.num_levels - 1)
            readout_frequency_assignment[i] = float(self.readout_frequency_options[freq_idx])

        # 验证解的有效性
        is_valid = self.validate_solution({'readout_frequencies': readout_frequency_assignment})

        return {
            'readout_frequencies': readout_frequency_assignment,
            'compute_frequencies': self.compute_frequencies,
            'bitstring': clean_bits,
            'is_valid': is_valid,
            'num_violations': self._count_violations(readout_frequency_assignment)
        }

    def validate_solution(self, solution: Dict[str, Any]) -> bool:
        """验证读出频率分配方案的有效性

        Parameters
        ----------
        solution : dict
            包含'readout_frequencies'字典的解

        Returns
        -------
        bool
            是否满足所有约束
        """
        readout_freqs = solution.get('readout_frequencies', {})

        if len(readout_freqs) != self.num_qubits:
            return False

        # 1. 检查频率范围
        f_min, f_max = self.problem_params['readout_freq_range']
        for freq in readout_freqs.values():
            if freq < f_min or freq > f_max:
                return False

        # 2. 检查相邻读出谐振器频率间隔
        for i, j in self.topology.edges():
            freq_diff = abs(readout_freqs[i] - readout_freqs[j])
            if freq_diff < self.min_readout_separation:
                return False

        # 3. 如果有计算频率，检查读出-计算频率间隔
        if self.compute_frequencies:
            for qubit_id, compute_freq in self.compute_frequencies.items():
                if qubit_id in readout_freqs:
                    freq_diff = abs(readout_freqs[qubit_id] - compute_freq)
                    if freq_diff < self.min_readout_compute_separation:
                        return False

        return True

    def _count_violations(self, readout_freqs: Dict[int, float]) -> int:
        """统计约束违反数量"""
        violations = 0

        # 相邻读出谐振器频率间隔违反
        for i, j in self.topology.edges():
            freq_diff = abs(readout_freqs[i] - readout_freqs[j])
            if freq_diff < self.min_readout_separation:
                violations += 1

        # 读出-计算频率间隔违反
        if self.compute_frequencies:
            for qubit_id, compute_freq in self.compute_frequencies.items():
                if qubit_id in readout_freqs:
                    freq_diff = abs(readout_freqs[qubit_id] - compute_freq)
                    if freq_diff < self.min_readout_compute_separation:
                        violations += 1

        return violations

    def evaluate_solution(self, solution: Dict[str, Any]) -> float:
        """评估解的质量"""
        readout_freqs = solution.get('readout_frequencies', {})

        # 计算总违反数
        violations = self._count_violations(readout_freqs)

        # 计算频率多样性
        unique_freqs = len(set(readout_freqs.values()))
        diversity_score = unique_freqs / self.num_qubits

        # 综合得分
        score = violations * 100 - diversity_score * 10

        return float(score)

    def get_frequency_statistics(
        self,
        solution: Dict[str, Any]
    ) -> Dict[str, Any]:
        """获取频率分配的统计信息"""
        readout_freqs = solution.get('readout_frequencies', {})
        readout_values = list(readout_freqs.values())

        # 计算读出频率间隔
        readout_diffs = []
        for i, j in self.topology.edges():
            readout_diffs.append(abs(readout_freqs[i] - readout_freqs[j]))

        stats = {
            'mean_readout_frequency': np.mean(readout_values),
            'std_readout_frequency': np.std(readout_values),
            'num_unique_readout_frequencies': len(set(readout_values)),
            'min_readout_difference': np.min(readout_diffs) if readout_diffs else 0.0,
            'mean_readout_difference': np.mean(readout_diffs) if readout_diffs else 0.0,
        }

        # 如果有计算频率，添加相关统计
        if self.compute_frequencies:
            readout_compute_diffs = []
            for qubit_id, compute_freq in self.compute_frequencies.items():
                if qubit_id in readout_freqs:
                    readout_compute_diffs.append(
                        abs(readout_freqs[qubit_id] - compute_freq)
                    )

            if readout_compute_diffs:
                stats['min_readout_compute_separation'] = np.min(readout_compute_diffs)
                stats['mean_readout_compute_separation'] = np.mean(readout_compute_diffs)

        # 约束满足率
        total_constraints = len(self.topology.edges())
        if self.compute_frequencies:
            total_constraints += len(self.compute_frequencies)

        stats['constraint_satisfaction_rate'] = (
            1.0 - (self._count_violations(readout_freqs) / total_constraints)
            if total_constraints > 0 else 1.0
        )

        return stats


class JointFrequencyAllocationQUBO(GraphBasedFormulation):
    """计算频率和读出频率联合优化QUBO建模类

    同时优化计算频率和读出频率，考虑所有约束：
    1. 计算-计算频率分离
    2. 读出-读出频率分离
    3. 读出-计算频率分离

    Parameters
    ----------
    topology : nx.Graph
        量子比特连接拓扑图
    compute_freq_range : tuple of float
        计算频率范围
    readout_freq_range : tuple of float
        读出频率范围
    num_levels : int, optional
        频率离散化层数
    min_compute_separation : float, optional
        相邻计算频率最小间隔
    min_readout_separation : float, optional
        相邻读出频率最小间隔
    min_readout_compute_separation : float, optional
        读出-计算最小间隔
    penalty_weight : float, optional
        约束违反惩罚权重

    Notes
    -----
    这是一个更复杂的优化问题，需要2n个变量（n个用于计算频率，n个用于读出频率）。
    为了简化，我们使用两个独立的变量集。
    """

    def __init__(
        self,
        topology: nx.Graph,
        compute_freq_range: Tuple[float, float],
        readout_freq_range: Tuple[float, float],
        num_levels: int = 4,
        min_compute_separation: float = 0.2,
        min_readout_separation: float = 0.15,
        min_readout_compute_separation: float = 0.5,
        penalty_weight: float = 10.0
    ):
        problem_params = {
            'compute_freq_range': compute_freq_range,
            'readout_freq_range': readout_freq_range,
            'num_levels': num_levels,
            'min_compute_separation': min_compute_separation,
            'min_readout_separation': min_readout_separation,
            'min_readout_compute_separation': min_readout_compute_separation,
            'penalty_weight': penalty_weight
        }
        super().__init__(topology, problem_params)

        self.topology = topology
        self.num_qubits = len(topology.nodes())

        # 联合优化使用2n个变量
        self.num_variables = 2 * self.num_qubits

        # 保存参数（为后续实现预留）
        self.compute_freq_range = compute_freq_range
        self.readout_freq_range = readout_freq_range
        self.num_levels = num_levels
        self.min_compute_separation = min_compute_separation
        self.min_readout_separation = min_readout_separation
        self.min_readout_compute_separation = min_readout_compute_separation
        self.penalty_weight = penalty_weight

    def build_qubo(self) -> NDArray[np.float64]:
        """构建联合优化QUBO矩阵

        Notes
        -----
        这是一个复杂的实现，当前版本采用简化策略：
        前n个变量用于计算频率，后n个变量用于读出频率
        """
        n = self.num_qubits
        Q = np.zeros((2*n, 2*n))

        # 1. 计算频率约束（前n个变量）
        for i, j in self.topology.edges():
            Q[i, j] += self.penalty_weight
            Q[j, i] += self.penalty_weight

        # 2. 读出频率约束（后n个变量）
        for i, j in self.topology.edges():
            Q[n+i, n+j] += self.penalty_weight
            Q[n+j, n+i] += self.penalty_weight

        # 3. 读出-计算分离约束（交叉项）
        for i in range(n):
            # 如果计算和读出选择相同的频率组，添加惩罚
            Q[i, n+i] += self.penalty_weight * 2
            Q[n+i, i] += self.penalty_weight * 2

        # 4. 偏置项
        for i in range(2*n):
            Q[i, i] -= 1.0

        self.qubo_matrix = Q
        return Q

    def decode_solution(self, bitstring: str) -> Dict[str, Any]:
        """解码联合优化解"""
        clean_bits = bitstring.replace(' ', '')

        if len(clean_bits) != 2 * self.num_qubits:
            raise ValueError(
                f"Expected bitstring length {2*self.num_qubits}, got {len(clean_bits)}"
            )

        # 前n位是计算频率，后n位是读出频率
        compute_bits = clean_bits[:self.num_qubits]
        readout_bits = clean_bits[self.num_qubits:]

        # 解码计算频率（简化实现）
        compute_freqs = {}
        readout_freqs = {}

        for i in range(self.num_qubits):
            # 这里需要实际的解码逻辑，当前为占位符
            compute_freqs[i] = 5.0 + i * 0.2
            readout_freqs[i] = 7.0 + i * 0.2

        return {
            'compute_frequencies': compute_freqs,
            'readout_frequencies': readout_freqs,
            'bitstring': clean_bits,
            'is_valid': True  # 占位符
        }

    def validate_solution(self, solution: Dict[str, Any]) -> bool:
        """验证联合优化解的有效性"""
        # 占位符实现
        return True


if __name__ == "__main__":
    print("ReadoutFrequencyAllocation模块加载成功!")
    print("可用类:")
    print("  - ReadoutFrequencyAllocationQUBO")
    print("  - JointFrequencyAllocationQUBO (实验性)")
