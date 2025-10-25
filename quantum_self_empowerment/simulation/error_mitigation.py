"""误差缓解技术模块

提供量子计算中常用的误差缓解技术实现。

Classes:
    ZeroNoiseExtrapolation: 零噪声外推法
    MeasurementErrorMitigation: 测量误差缓解
    DynamicalDecoupling: 动力学解耦

Functions:
    apply_zne: 应用零噪声外推
    mitigate_readout_error: 缓解读取错误
    add_dynamical_decoupling: 添加动力学解耦序列

Author: [待填写]
Created: 2025-10-17
"""

from typing import Dict, List, Optional, Tuple, Callable, Union, Any
from dataclasses import dataclass, field
from abc import ABC, abstractmethod
import numpy as np
from numpy.typing import NDArray
import warnings

from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator
from qiskit.result import Result
from scipy.optimize import curve_fit
from scipy.linalg import inv


@dataclass
class ZNEResult:
    """零噪声外推结果

    Attributes
    ----------
    mitigated_expectation : float
        缓解后的期望值
    noisy_expectations : list of float
        不同噪声水平的期望值
    noise_factors : list of float
        噪声放大因子
    fit_params : dict
        拟合参数
    extrapolation_model : str
        外推模型类型
    """
    mitigated_expectation: float
    noisy_expectations: List[float]
    noise_factors: List[float]
    fit_params: Dict[str, Any]
    extrapolation_model: str = "exponential"


class ZeroNoiseExtrapolation:
    """零噪声外推法（Zero-Noise Extrapolation, ZNE）

    通过人为放大噪声，测量不同噪声水平下的期望值，
    然后外推到零噪声极限。

    Parameters
    ----------
    noise_factors : list of float, optional
        噪声放大因子，默认为 [1, 2, 3]
    extrapolation_model : str, optional
        外推模型，可选:
        - 'linear': 线性外推
        - 'exponential': 指数外推（默认）
        - 'polynomial': 多项式外推
    polynomial_degree : int, optional
        多项式外推的阶数，默认2
    verbose : bool, optional
        详细输出

    Examples
    --------
    >>> zne = ZeroNoiseExtrapolation(noise_factors=[1, 2, 3])
    >>> result = zne.apply(circuit, observable, backend)
    >>> print(f"Mitigated expectation: {result.mitigated_expectation}")

    Notes
    -----
    ZNE基本原理：

    1. 测量不同噪声水平 :math:`\\lambda` 下的期望值 :math:`E(\\lambda)`
    2. 拟合模型 :math:`E(\\lambda) = f(\\lambda)`
    3. 外推到零噪声 :math:`E_{ideal} = f(0)`

    常用外推模型：
    - 线性: :math:`E(\\lambda) = a + b\\lambda`
    - 指数: :math:`E(\\lambda) = a + b\\exp(-c\\lambda)`

    References
    ----------
    .. [1] Temme, K., et al. "Error mitigation for short-depth quantum circuits."
           Phys. Rev. Lett. 119, 180509 (2017)
    """

    def __init__(
        self,
        noise_factors: List[float] = None,
        extrapolation_model: str = "exponential",
        polynomial_degree: int = 2,
        verbose: bool = False
    ):
        self.noise_factors = noise_factors or [1.0, 2.0, 3.0]
        self.extrapolation_model = extrapolation_model
        self.polynomial_degree = polynomial_degree
        self.verbose = verbose

        # 验证参数
        if min(self.noise_factors) < 1.0:
            raise ValueError("Noise factors must be >= 1.0")

        if extrapolation_model not in ['linear', 'exponential', 'polynomial']:
            raise ValueError(
                f"Unknown extrapolation model: {extrapolation_model}. "
                f"Choose from: linear, exponential, polynomial"
            )

    def amplify_noise(
        self,
        circuit: QuantumCircuit,
        noise_factor: float
    ) -> QuantumCircuit:
        """通过门折叠放大噪声

        Parameters
        ----------
        circuit : QuantumCircuit
            原始电路
        noise_factor : float
            噪声放大因子（必须为奇数倍，如1, 3, 5...）

        Returns
        -------
        QuantumCircuit
            噪声放大的电路

        Notes
        -----
        门折叠方法：将每个门 G 替换为 G·G†·G，实现噪声放大约3倍。
        对于任意放大因子，通过部分折叠实现。
        """
        if noise_factor == 1.0:
            return circuit.copy()

        # 简化实现：整体电路折叠
        # G -> G·G†·G (噪声放大约3倍)
        # 对于其他因子，使用近似方法

        num_folds = int((noise_factor - 1) / 2)

        if num_folds < 1:
            warnings.warn(
                f"Noise factor {noise_factor} < 3, using original circuit"
            )
            return circuit.copy()

        # 创建放大噪声的电路
        amplified_circuit = QuantumCircuit(circuit.num_qubits, circuit.num_clbits)

        # 复制原电路
        amplified_circuit.compose(circuit.remove_final_measurements(inplace=False), inplace=True)

        # 添加折叠（G†·G重复）
        for _ in range(num_folds):
            # 添加逆电路
            amplified_circuit.compose(
                circuit.remove_final_measurements(inplace=False).inverse(),
                inplace=True
            )
            # 再添加正向电路
            amplified_circuit.compose(
                circuit.remove_final_measurements(inplace=False),
                inplace=True
            )

        # 添加测量
        amplified_circuit.measure_all()

        return amplified_circuit

    def _fit_model(
        self,
        noise_factors: NDArray,
        expectations: NDArray
    ) -> Tuple[Callable, Dict]:
        """拟合外推模型

        Parameters
        ----------
        noise_factors : NDArray
            噪声因子数组
        expectations : NDArray
            期望值数组

        Returns
        -------
        model_func : Callable
            拟合的模型函数
        params : dict
            拟合参数
        """
        if self.extrapolation_model == 'linear':
            # 线性拟合: E = a + b*x
            def model(x, a, b):
                return a + b * x

            popt, _ = curve_fit(model, noise_factors, expectations)
            return model, {'a': popt[0], 'b': popt[1]}

        elif self.extrapolation_model == 'exponential':
            # 指数拟合: E = a + b*exp(-c*x)
            def model(x, a, b, c):
                return a + b * np.exp(-c * x)

            try:
                popt, _ = curve_fit(
                    model,
                    noise_factors,
                    expectations,
                    p0=[expectations[-1], expectations[0] - expectations[-1], 1.0],
                    maxfev=5000
                )
                return model, {'a': popt[0], 'b': popt[1], 'c': popt[2]}
            except RuntimeError:
                warnings.warn("Exponential fit failed, falling back to linear")
                return self._fit_model(noise_factors, expectations)

        elif self.extrapolation_model == 'polynomial':
            # 多项式拟合
            coeffs = np.polyfit(noise_factors, expectations, self.polynomial_degree)

            def model(x):
                return np.polyval(coeffs, x)

            param_dict = {f'c{i}': coeffs[i] for i in range(len(coeffs))}
            return model, param_dict

    def apply(
        self,
        circuit: QuantumCircuit,
        cost_function: Callable[[Dict[str, int]], float],
        backend: Any,
        shots: int = 1024
    ) -> ZNEResult:
        """应用零噪声外推

        Parameters
        ----------
        circuit : QuantumCircuit
            量子电路
        cost_function : Callable
            成本函数，接受测量结果counts，返回期望值
        backend : Backend
            量子后端
        shots : int, optional
            测量次数

        Returns
        -------
        ZNEResult
            缓解结果
        """
        if self.verbose:
            print("="*60)
            print("应用零噪声外推（ZNE）")
            print("="*60)
            print(f"噪声因子: {self.noise_factors}")
            print(f"外推模型: {self.extrapolation_model}")

        noisy_expectations = []

        # 对每个噪声因子运行电路
        for factor in self.noise_factors:
            # 放大噪声
            amplified_circuit = self.amplify_noise(circuit, factor)

            # 运行电路
            job = backend.run(amplified_circuit, shots=shots)
            result = job.result()
            counts = result.get_counts()

            # 计算期望值
            expectation = cost_function(counts)
            noisy_expectations.append(expectation)

            if self.verbose:
                print(f"  噪声因子 {factor:.1f}: 期望值 = {expectation:.6f}")

        # 拟合模型并外推到零噪声
        noise_factors_array = np.array(self.noise_factors)
        expectations_array = np.array(noisy_expectations)

        model_func, fit_params = self._fit_model(noise_factors_array, expectations_array)

        # 外推到零噪声
        if self.extrapolation_model in ['linear', 'exponential']:
            mitigated_expectation = model_func(0, **fit_params)
        else:  # polynomial
            mitigated_expectation = model_func(0)

        if self.verbose:
            print(f"\n拟合参数: {fit_params}")
            print(f"零噪声外推值: {mitigated_expectation:.6f}")
            print(f"改进: {abs(mitigated_expectation - noisy_expectations[0]):.6f}")
            print("="*60)

        return ZNEResult(
            mitigated_expectation=mitigated_expectation,
            noisy_expectations=noisy_expectations,
            noise_factors=self.noise_factors,
            fit_params=fit_params,
            extrapolation_model=self.extrapolation_model
        )


class MeasurementErrorMitigation:
    """测量误差缓解

    通过校准矩阵修正测量错误。

    Parameters
    ----------
    backend : Backend
        量子后端
    qubits : list of int, optional
        要校准的量子比特（None表示所有量子比特）
    shots : int, optional
        校准测量次数，默认1024
    verbose : bool, optional
        详细输出

    Examples
    --------
    >>> mem = MeasurementErrorMitigation(backend, qubits=[0, 1, 2])
    >>> mem.calibrate()
    >>> mitigated_counts = mem.apply(noisy_counts)

    Notes
    -----
    测量误差缓解流程：

    1. **校准阶段**：制备计算基态 |00...0⟩, |00...1⟩, ..., |11...1⟩
    2. **测量**：记录实际测量结果，构建校准矩阵 M
    3. **应用**：使用 M^{-1} 修正测量结果

    校准矩阵元素：:math:`M_{ij}` = P(测量到i | 真实态j)
    """

    def __init__(
        self,
        backend: Any,
        qubits: Optional[List[int]] = None,
        shots: int = 1024,
        verbose: bool = False
    ):
        self.backend = backend
        self.qubits = qubits
        self.shots = shots
        self.verbose = verbose
        self.calibration_matrix: Optional[NDArray] = None

    def calibrate(self):
        """执行校准测量

        构建测量误差校准矩阵。
        """
        if self.verbose:
            print("="*60)
            print("测量误差校准")
            print("="*60)

        # 确定量子比特数量
        if self.qubits is None:
            # 假设使用所有可用量子比特（简化实现）
            num_qubits = 4  # 默认值，实际应从backend获取
        else:
            num_qubits = len(self.qubits)

        num_states = 2 ** num_qubits
        calibration_matrix = np.zeros((num_states, num_states))

        # 为每个计算基态制备电路并测量
        for state_idx in range(num_states):
            # 创建制备电路
            qc = QuantumCircuit(num_qubits, num_qubits)

            # 制备计算基态（二进制表示）
            state_binary = format(state_idx, f'0{num_qubits}b')
            for i, bit in enumerate(state_binary):
                if bit == '1':
                    qc.x(i)

            # 测量
            qc.measure_all()

            # 运行
            job = self.backend.run(qc, shots=self.shots)
            result = job.result()
            counts = result.get_counts()

            # 统计每个测量结果的概率
            for measured_state, count in counts.items():
                # 清理比特串（移除空格）
                clean_bits = measured_state.replace(' ', '')

                # 检查比特串长度是否匹配
                if len(clean_bits) != num_qubits:
                    # Qiskit可能返回不同长度的比特串，跳过不匹配的
                    warnings.warn(
                        f"Skipping measurement result with unexpected length: "
                        f"{len(clean_bits)} (expected {num_qubits})"
                    )
                    continue

                measured_idx = int(clean_bits, 2)

                # 边界检查
                if measured_idx >= num_states:
                    warnings.warn(
                        f"Measured index {measured_idx} out of bounds for "
                        f"calibration matrix size {num_states}"
                    )
                    continue

                calibration_matrix[measured_idx, state_idx] = count / self.shots

            if self.verbose and state_idx % 4 == 0:
                print(f"  已校准 {state_idx+1}/{num_states} 个基态")

        self.calibration_matrix = calibration_matrix

        if self.verbose:
            print(f"✓ 校准完成！矩阵形状: {calibration_matrix.shape}")
            print(f"  校准矩阵对角线均值: {np.mean(np.diag(calibration_matrix)):.4f}")
            print("="*60)

    def apply(
        self,
        noisy_counts: Dict[str, int]
    ) -> Dict[str, float]:
        """应用误差缓解

        Parameters
        ----------
        noisy_counts : dict
            含噪声的测量结果

        Returns
        -------
        dict
            缓解后的测量结果（可能包含负值或非整数）

        Raises
        ------
        RuntimeError
            如果尚未执行校准
        """
        if self.calibration_matrix is None:
            raise RuntimeError("Must run calibrate() before applying mitigation")

        # 将counts转换为概率向量
        total_shots = sum(noisy_counts.values())
        num_states = self.calibration_matrix.shape[0]
        num_qubits = int(np.log2(num_states))

        noisy_probs = np.zeros(num_states)
        for bitstring, count in noisy_counts.items():
            clean_bits = bitstring.replace(' ', '')

            # 检查长度并跳过不匹配的
            if len(clean_bits) != num_qubits:
                warnings.warn(
                    f"Skipping measurement with unexpected length: "
                    f"{len(clean_bits)} (expected {num_qubits})"
                )
                continue

            idx = int(clean_bits, 2)

            # 边界检查
            if idx >= num_states:
                warnings.warn(f"Index {idx} out of bounds, skipping")
                continue

            noisy_probs[idx] = count / total_shots

        # 应用逆校准矩阵
        try:
            inv_matrix = inv(self.calibration_matrix)
            mitigated_probs = inv_matrix @ noisy_probs
        except np.linalg.LinAlgError:
            warnings.warn("Calibration matrix singular, using pseudo-inverse")
            inv_matrix = np.linalg.pinv(self.calibration_matrix)
            mitigated_probs = inv_matrix @ noisy_probs

        # 转换回counts格式（保持浮点数）
        mitigated_counts = {}
        for idx, prob in enumerate(mitigated_probs):
            if abs(prob) > 1e-6:  # 忽略很小的值
                bitstring = format(idx, f'0{num_qubits}b')
                mitigated_counts[bitstring] = prob * total_shots

        if self.verbose:
            print(f"误差缓解完成:")
            print(f"  原始结果数: {len(noisy_counts)}")
            print(f"  缓解后结果数: {len(mitigated_counts)}")

        return mitigated_counts


class DynamicalDecoupling:
    """动力学解耦

    通过插入快速脉冲序列抑制退相干噪声。

    Parameters
    ----------
    sequence_type : str, optional
        解耦序列类型:
        - 'XY4': XY4序列（默认）
        - 'CPMG': Carr-Purcell-Meiboom-Gill序列
        - 'UDD': Uhrig动力学解耦
    num_pulses : int, optional
        脉冲数量，默认4
    verbose : bool, optional
        详细输出

    Examples
    --------
    >>> dd = DynamicalDecoupling(sequence_type='XY4', num_pulses=4)
    >>> protected_circuit = dd.apply(circuit, idle_qubits=[0, 1])

    Notes
    -----
    动力学解耦原理：

    在量子比特空闲期间插入快速脉冲，平均掉环境噪声的影响。

    常用序列：
    - **CPMG**: X-X-X-X (简单但有效)
    - **XY4**: X-Y-X-Y (抑制更多噪声)
    - **UDD**: 脉冲间隔非均匀分布（最优时间分布）

    References
    ----------
    .. [1] Viola, L., & Lloyd, S. "Dynamical suppression of decoherence in
           two-state quantum systems." Phys. Rev. A 58, 2733 (1998)
    """

    def __init__(
        self,
        sequence_type: str = 'XY4',
        num_pulses: int = 4,
        verbose: bool = False
    ):
        self.sequence_type = sequence_type
        self.num_pulses = num_pulses
        self.verbose = verbose

        valid_sequences = ['XY4', 'CPMG', 'UDD']
        if sequence_type not in valid_sequences:
            raise ValueError(
                f"Unknown sequence type: {sequence_type}. "
                f"Choose from: {valid_sequences}"
            )

    def get_pulse_sequence(self) -> List[str]:
        """获取脉冲序列

        Returns
        -------
        list of str
            脉冲门名称列表
        """
        if self.sequence_type == 'CPMG':
            # Carr-Purcell-Meiboom-Gill: 重复X门
            return ['x'] * self.num_pulses

        elif self.sequence_type == 'XY4':
            # XY4序列: X-Y-X-Y 重复
            base_sequence = ['x', 'y', 'x', 'y']
            full_sequence = base_sequence * (self.num_pulses // 4)
            # 添加剩余部分
            full_sequence += base_sequence[:self.num_pulses % 4]
            return full_sequence

        elif self.sequence_type == 'UDD':
            # Uhrig DD: X门，但间隔非均匀
            # 简化实现：仅返回X门列表
            return ['x'] * self.num_pulses

    def apply(
        self,
        circuit: QuantumCircuit,
        idle_qubits: Optional[List[int]] = None
    ) -> QuantumCircuit:
        """应用动力学解耦

        Parameters
        ----------
        circuit : QuantumCircuit
            原始电路
        idle_qubits : list of int, optional
            需要保护的空闲量子比特（None表示所有）

        Returns
        -------
        QuantumCircuit
            添加了解耦序列的电路

        Notes
        -----
        简化实现：在电路末尾添加解耦序列。
        完整实现应在每个空闲期间插入。
        """
        if idle_qubits is None:
            idle_qubits = list(range(circuit.num_qubits))

        protected_circuit = circuit.copy()
        pulse_sequence = self.get_pulse_sequence()

        # 在指定量子比特上添加解耦脉冲
        for qubit in idle_qubits:
            for pulse in pulse_sequence:
                if pulse == 'x':
                    protected_circuit.x(qubit)
                elif pulse == 'y':
                    protected_circuit.y(qubit)

        if self.verbose:
            print(f"✓ 动力学解耦已应用")
            print(f"  序列类型: {self.sequence_type}")
            print(f"  脉冲数量: {self.num_pulses}")
            print(f"  保护量子比特: {idle_qubits}")

        return protected_circuit


# ============ 便捷函数 ============

def apply_zne(
    circuit: QuantumCircuit,
    cost_function: Callable,
    backend: Any,
    noise_factors: List[float] = None,
    **kwargs
) -> ZNEResult:
    """便捷函数：应用零噪声外推

    Parameters
    ----------
    circuit : QuantumCircuit
        量子电路
    cost_function : Callable
        成本函数
    backend : Backend
        量子后端
    noise_factors : list of float, optional
        噪声因子
    **kwargs
        传递给ZeroNoiseExtrapolation的其他参数

    Returns
    -------
    ZNEResult
        缓解结果

    Examples
    --------
    >>> result = apply_zne(circuit, cost_func, backend, noise_factors=[1, 2, 3])
    """
    zne = ZeroNoiseExtrapolation(noise_factors=noise_factors, **kwargs)
    return zne.apply(circuit, cost_function, backend)


def mitigate_readout_error(
    backend: Any,
    noisy_counts: Dict[str, int],
    qubits: Optional[List[int]] = None,
    **kwargs
) -> Dict[str, float]:
    """便捷函数：缓解读取错误

    Parameters
    ----------
    backend : Backend
        量子后端
    noisy_counts : dict
        含噪声的测量结果
    qubits : list of int, optional
        量子比特列表
    **kwargs
        传递给MeasurementErrorMitigation的其他参数

    Returns
    -------
    dict
        缓解后的测量结果

    Examples
    --------
    >>> mitigated = mitigate_readout_error(backend, noisy_counts, qubits=[0,1,2])
    """
    mem = MeasurementErrorMitigation(backend, qubits=qubits, **kwargs)
    mem.calibrate()
    return mem.apply(noisy_counts)


def add_dynamical_decoupling(
    circuit: QuantumCircuit,
    sequence_type: str = 'XY4',
    **kwargs
) -> QuantumCircuit:
    """便捷函数：添加动力学解耦

    Parameters
    ----------
    circuit : QuantumCircuit
        原始电路
    sequence_type : str, optional
        解耦序列类型
    **kwargs
        传递给DynamicalDecoupling的其他参数

    Returns
    -------
    QuantumCircuit
        添加了解耦的电路

    Examples
    --------
    >>> protected = add_dynamical_decoupling(circuit, sequence_type='XY4')
    """
    dd = DynamicalDecoupling(sequence_type=sequence_type, **kwargs)
    return dd.apply(circuit)


if __name__ == "__main__":
    print("误差缓解模块加载成功!")
    print("\n可用误差缓解技术:")
    print("  - ZeroNoiseExtrapolation (ZNE): 零噪声外推")
    print("  - MeasurementErrorMitigation: 测量误差缓解")
    print("  - DynamicalDecoupling: 动力学解耦")
    print("\n便捷函数:")
    print("  - apply_zne: 应用ZNE")
    print("  - mitigate_readout_error: 缓解读取错误")
    print("  - add_dynamical_decoupling: 添加动力学解耦")
