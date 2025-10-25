"""量子噪声模型模块

提供各种量子噪声模型的实现，用于模拟真实量子硬件的噪声特性。

Classes:
    NoiseModel: 噪声模型基类
    DepolarizingNoise: 去极化噪声
    AmplitudeDampingNoise: 振幅阻尼噪声
    ThermalRelaxationNoise: 热弛豫噪声
    ReadoutNoise: 读取噪声
    SuperconductingNoiseModel: 超导量子比特噪声模型

Functions:
    create_noise_model: 便捷函数，创建噪声模型

Author: [待填写]
Created: 2025-10-17
"""

from typing import Dict, List, Optional, Tuple, Union, Any
from dataclasses import dataclass, field
from abc import ABC, abstractmethod
import numpy as np
from numpy.typing import NDArray
import warnings

from qiskit_aer.noise import NoiseModel as QiskitNoiseModel
from qiskit_aer.noise import (
    depolarizing_error,
    amplitude_damping_error,
    thermal_relaxation_error,
    ReadoutError,
    pauli_error
)
from qiskit.providers.fake_provider import FakeBackend


@dataclass
class NoiseParameters:
    """噪声参数类

    Attributes
    ----------
    gate_error_1q : float
        单量子比特门错误率
    gate_error_2q : float
        双量子比特门错误率
    t1 : float
        T1弛豫时间（微秒）
    t2 : float
        T2退相干时间（微秒）
    gate_time_1q : float
        单量子比特门时间（纳秒）
    gate_time_2q : float
        双量子比特门时间（纳秒）
    readout_error : float
        读取错误率
    thermal_population : float
        热激发态占据数
    """
    gate_error_1q: float = 0.001
    gate_error_2q: float = 0.01
    t1: float = 50.0  # μs
    t2: float = 70.0  # μs
    gate_time_1q: float = 50.0  # ns
    gate_time_2q: float = 300.0  # ns
    readout_error: float = 0.02
    thermal_population: float = 0.0

    def __post_init__(self):
        """验证参数有效性"""
        if self.gate_error_1q < 0 or self.gate_error_1q > 1:
            raise ValueError(f"gate_error_1q must be in [0, 1], got {self.gate_error_1q}")
        if self.gate_error_2q < 0 or self.gate_error_2q > 1:
            raise ValueError(f"gate_error_2q must be in [0, 1], got {self.gate_error_2q}")
        if self.t1 <= 0:
            raise ValueError(f"t1 must be positive, got {self.t1}")
        if self.t2 <= 0:
            raise ValueError(f"t2 must be positive, got {self.t2}")
        if self.t2 > 2 * self.t1:
            warnings.warn(
                f"T2 ({self.t2}) should not exceed 2*T1 ({2*self.t1}). "
                f"Adjusting T2 to 2*T1."
            )
            self.t2 = 2 * self.t1
        if self.readout_error < 0 or self.readout_error > 1:
            raise ValueError(f"readout_error must be in [0, 1], got {self.readout_error}")


class NoiseModel(ABC):
    """噪声模型基类

    定义噪声模型的通用接口。

    Parameters
    ----------
    params : NoiseParameters
        噪声参数
    verbose : bool, optional
        是否显示详细信息，默认False

    Examples
    --------
    >>> params = NoiseParameters(gate_error_1q=0.001, gate_error_2q=0.01)
    >>> noise_model = DepolarizingNoise(params)
    >>> qiskit_noise = noise_model.build()
    """

    def __init__(self, params: NoiseParameters, verbose: bool = False):
        self.params = params
        self.verbose = verbose
        self._qiskit_noise_model: Optional[QiskitNoiseModel] = None

    @abstractmethod
    def build(self) -> QiskitNoiseModel:
        """构建Qiskit噪声模型

        Returns
        -------
        QiskitNoiseModel
            Qiskit噪声模型对象
        """
        pass

    def get_noise_model(self) -> QiskitNoiseModel:
        """获取噪声模型（缓存版本）

        Returns
        -------
        QiskitNoiseModel
            噪声模型
        """
        if self._qiskit_noise_model is None:
            self._qiskit_noise_model = self.build()
        return self._qiskit_noise_model

    def summary(self) -> str:
        """生成噪声模型摘要

        Returns
        -------
        str
            噪声模型参数摘要
        """
        summary = f"{self.__class__.__name__} Parameters:\n"
        summary += f"  Single-qubit gate error: {self.params.gate_error_1q:.4f}\n"
        summary += f"  Two-qubit gate error: {self.params.gate_error_2q:.4f}\n"
        summary += f"  T1 relaxation time: {self.params.t1:.1f} μs\n"
        summary += f"  T2 dephasing time: {self.params.t2:.1f} μs\n"
        summary += f"  Readout error: {self.params.readout_error:.4f}\n"
        return summary


class DepolarizingNoise(NoiseModel):
    """去极化噪声模型

    模拟门操作后的去极化错误。去极化信道将量子态以一定概率
    转变为完全混合态。

    Parameters
    ----------
    params : NoiseParameters
        噪声参数
    apply_to_all_gates : bool, optional
        是否应用于所有门，默认True
    verbose : bool, optional
        详细输出

    Examples
    --------
    >>> params = NoiseParameters(gate_error_1q=0.001, gate_error_2q=0.01)
    >>> noise = DepolarizingNoise(params)
    >>> qiskit_noise = noise.build()

    Notes
    -----
    去极化信道定义为：

    .. math::
        \\mathcal{E}(\\rho) = (1-p)\\rho + \\frac{p}{d^2-1}\\sum_{i=1}^{d^2-1} P_i \\rho P_i^\\dagger

    其中 :math:`p` 是错误概率，:math:`P_i` 是Pauli算符。
    """

    def __init__(
        self,
        params: NoiseParameters,
        apply_to_all_gates: bool = True,
        verbose: bool = False
    ):
        super().__init__(params, verbose)
        self.apply_to_all_gates = apply_to_all_gates

    def build(self) -> QiskitNoiseModel:
        """构建去极化噪声模型"""
        noise_model = QiskitNoiseModel()

        # 单量子比特去极化错误
        error_1q = depolarizing_error(self.params.gate_error_1q, 1)

        # 双量子比特去极化错误
        error_2q = depolarizing_error(self.params.gate_error_2q, 2)

        if self.apply_to_all_gates:
            # 应用于所有门
            noise_model.add_all_qubit_quantum_error(error_1q, ['h', 'rx', 'ry', 'rz', 'x', 'y', 'z'])
            noise_model.add_all_qubit_quantum_error(error_2q, ['cx', 'cz', 'rzz'])
        else:
            # 仅应用于基本门
            noise_model.add_all_qubit_quantum_error(error_1q, ['h', 'x'])
            noise_model.add_all_qubit_quantum_error(error_2q, ['cx'])

        if self.verbose:
            print(f"✓ 去极化噪声模型已构建")
            print(f"  单比特门错误率: {self.params.gate_error_1q:.4f}")
            print(f"  双比特门错误率: {self.params.gate_error_2q:.4f}")

        return noise_model


class AmplitudeDampingNoise(NoiseModel):
    """振幅阻尼噪声模型

    模拟能量弛豫过程，量子比特从 |1⟩ 态衰减到 |0⟩ 态。

    Parameters
    ----------
    params : NoiseParameters
        噪声参数
    verbose : bool, optional
        详细输出

    Examples
    --------
    >>> params = NoiseParameters(t1=50.0, gate_time_1q=50.0)
    >>> noise = AmplitudeDampingNoise(params)
    >>> qiskit_noise = noise.build()

    Notes
    -----
    振幅阻尼参数 :math:`\\gamma` 与T1的关系：

    .. math::
        \\gamma = 1 - \\exp(-t_{gate}/T_1)

    其中 :math:`t_{gate}` 是门操作时间。
    """

    def build(self) -> QiskitNoiseModel:
        """构建振幅阻尼噪声模型"""
        noise_model = QiskitNoiseModel()

        # 计算阻尼参数
        # gamma = 1 - exp(-t_gate / T1)
        # 单位转换: T1 (μs), gate_time (ns)
        gamma_1q = 1 - np.exp(-self.params.gate_time_1q * 1e-3 / self.params.t1)
        gamma_2q = 1 - np.exp(-self.params.gate_time_2q * 1e-3 / self.params.t1)

        # 单量子比特振幅阻尼错误
        error_1q = amplitude_damping_error(gamma_1q)

        # 双量子比特振幅阻尼错误（作用于两个量子比特）
        error_2q = amplitude_damping_error(gamma_2q).tensor(
            amplitude_damping_error(gamma_2q)
        )

        # 添加到噪声模型
        noise_model.add_all_qubit_quantum_error(error_1q, ['h', 'rx', 'ry', 'rz', 'x', 'y', 'z'])
        noise_model.add_all_qubit_quantum_error(error_2q, ['cx', 'cz', 'rzz'])

        if self.verbose:
            print(f"✓ 振幅阻尼噪声模型已构建")
            print(f"  T1弛豫时间: {self.params.t1:.1f} μs")
            print(f"  单比特门阻尼参数: {gamma_1q:.6f}")
            print(f"  双比特门阻尼参数: {gamma_2q:.6f}")

        return noise_model


class ThermalRelaxationNoise(NoiseModel):
    """热弛豫噪声模型

    综合模拟T1能量弛豫和T2退相干过程，是最接近真实超导
    量子比特的噪声模型。

    Parameters
    ----------
    params : NoiseParameters
        噪声参数
    verbose : bool, optional
        详细输出

    Examples
    --------
    >>> params = NoiseParameters(t1=50.0, t2=70.0, gate_time_1q=50.0)
    >>> noise = ThermalRelaxationNoise(params)
    >>> qiskit_noise = noise.build()

    Notes
    -----
    热弛豫信道同时考虑：
    - T1过程：能量弛豫（振幅阻尼）
    - T2过程：退相干（相位阻尼）

    要求 :math:`T_2 \\leq 2T_1`（物理约束）。
    """

    def build(self) -> QiskitNoiseModel:
        """构建热弛豫噪声模型"""
        noise_model = QiskitNoiseModel()

        # 单量子比特热弛豫错误
        error_1q = thermal_relaxation_error(
            t1=self.params.t1 * 1e3,  # 转换为ns
            t2=self.params.t2 * 1e3,
            time=self.params.gate_time_1q,
            excited_state_population=self.params.thermal_population
        )

        # 双量子比特热弛豫错误
        error_2q = thermal_relaxation_error(
            t1=self.params.t1 * 1e3,
            t2=self.params.t2 * 1e3,
            time=self.params.gate_time_2q,
            excited_state_population=self.params.thermal_population
        ).tensor(
            thermal_relaxation_error(
                t1=self.params.t1 * 1e3,
                t2=self.params.t2 * 1e3,
                time=self.params.gate_time_2q,
                excited_state_population=self.params.thermal_population
            )
        )

        # 添加到噪声模型
        noise_model.add_all_qubit_quantum_error(error_1q, ['h', 'rx', 'ry', 'rz', 'x', 'y', 'z'])
        noise_model.add_all_qubit_quantum_error(error_2q, ['cx', 'cz', 'rzz'])

        if self.verbose:
            print(f"✓ 热弛豫噪声模型已构建")
            print(f"  T1: {self.params.t1:.1f} μs, T2: {self.params.t2:.1f} μs")
            print(f"  单比特门时间: {self.params.gate_time_1q:.0f} ns")
            print(f"  双比特门时间: {self.params.gate_time_2q:.0f} ns")

        return noise_model


class ReadoutNoise(NoiseModel):
    """读取噪声模型

    模拟量子态测量时的错误，包括 |0⟩ 误读为 |1⟩ 和 |1⟩ 误读为 |0⟩。

    Parameters
    ----------
    params : NoiseParameters
        噪声参数
    asymmetric : bool, optional
        是否使用非对称读取错误，默认False
    p0given1 : float, optional
        |1⟩ 误读为 |0⟩ 的概率（仅在asymmetric=True时使用）
    p1given0 : float, optional
        |0⟩ 误读为 |1⟩ 的概率（仅在asymmetric=True时使用）
    verbose : bool, optional
        详细输出

    Examples
    --------
    >>> params = NoiseParameters(readout_error=0.02)
    >>> noise = ReadoutNoise(params)
    >>> qiskit_noise = noise.build()

    >>> # 非对称读取错误
    >>> noise_asym = ReadoutNoise(params, asymmetric=True, p0given1=0.05, p1given0=0.01)

    Notes
    -----
    读取错误矩阵形式：

    .. math::
        M = \\begin{pmatrix}
        1-p_{1|0} & p_{0|1} \\\\
        p_{1|0} & 1-p_{0|1}
        \\end{pmatrix}

    其中 :math:`p_{i|j}` 表示态 :math:`|j\\rangle` 被测量为 :math:`|i\\rangle` 的概率。
    """

    def __init__(
        self,
        params: NoiseParameters,
        asymmetric: bool = False,
        p0given1: float = None,
        p1given0: float = None,
        verbose: bool = False
    ):
        super().__init__(params, verbose)
        self.asymmetric = asymmetric
        self.p0given1 = p0given1 if p0given1 is not None else params.readout_error
        self.p1given0 = p1given0 if p1given0 is not None else params.readout_error / 2

    def build(self) -> QiskitNoiseModel:
        """构建读取噪声模型"""
        noise_model = QiskitNoiseModel()

        if self.asymmetric:
            # 非对称读取错误
            # 概率矩阵: [[P(0|0), P(0|1)], [P(1|0), P(1|1)]]
            readout_error_matrix = [
                [1 - self.p1given0, self.p0given1],
                [self.p1given0, 1 - self.p0given1]
            ]
        else:
            # 对称读取错误
            p_error = self.params.readout_error
            readout_error_matrix = [
                [1 - p_error, p_error],
                [p_error, 1 - p_error]
            ]

        # 创建读取错误
        readout_error = ReadoutError(readout_error_matrix)

        # 添加到噪声模型
        noise_model.add_all_qubit_readout_error(readout_error)

        if self.verbose:
            print(f"✓ 读取噪声模型已构建")
            if self.asymmetric:
                print(f"  P(0|1): {self.p0given1:.4f}")
                print(f"  P(1|0): {self.p1given0:.4f}")
            else:
                print(f"  对称读取错误率: {self.params.readout_error:.4f}")

        return noise_model


class SuperconductingNoiseModel(NoiseModel):
    """超导量子比特综合噪声模型

    结合多种噪声源，模拟真实超导量子芯片的噪声特性：
    - 热弛豫噪声（T1, T2）
    - 门错误（去极化）
    - 读取错误

    Parameters
    ----------
    params : NoiseParameters
        噪声参数
    include_thermal_relaxation : bool, optional
        是否包含热弛豫噪声，默认True
    include_gate_errors : bool, optional
        是否包含额外的门错误，默认True
    include_readout_errors : bool, optional
        是否包含读取错误，默认True
    verbose : bool, optional
        详细输出

    Examples
    --------
    >>> # 典型超导量子比特参数
    >>> params = NoiseParameters(
    ...     gate_error_1q=0.001,
    ...     gate_error_2q=0.01,
    ...     t1=50.0,
    ...     t2=70.0,
    ...     gate_time_1q=50.0,
    ...     gate_time_2q=300.0,
    ...     readout_error=0.02
    ... )
    >>> noise_model = SuperconductingNoiseModel(params)
    >>> qiskit_noise = noise_model.build()

    Notes
    -----
    此模型综合了多种噪声源，更接近真实量子硬件的表现。
    适用于评估算法在真实设备上的性能。
    """

    def __init__(
        self,
        params: NoiseParameters,
        include_thermal_relaxation: bool = True,
        include_gate_errors: bool = True,
        include_readout_errors: bool = True,
        verbose: bool = False
    ):
        super().__init__(params, verbose)
        self.include_thermal_relaxation = include_thermal_relaxation
        self.include_gate_errors = include_gate_errors
        self.include_readout_errors = include_readout_errors

    def build(self) -> QiskitNoiseModel:
        """构建综合噪声模型"""
        noise_model = QiskitNoiseModel()

        if self.verbose:
            print("="*60)
            print("构建超导量子比特综合噪声模型")
            print("="*60)

        # 1. 热弛豫噪声（基础噪声）
        if self.include_thermal_relaxation:
            error_1q_thermal = thermal_relaxation_error(
                t1=self.params.t1 * 1e3,
                t2=self.params.t2 * 1e3,
                time=self.params.gate_time_1q,
                excited_state_population=self.params.thermal_population
            )

            error_2q_thermal = thermal_relaxation_error(
                t1=self.params.t1 * 1e3,
                t2=self.params.t2 * 1e3,
                time=self.params.gate_time_2q,
                excited_state_population=self.params.thermal_population
            ).tensor(
                thermal_relaxation_error(
                    t1=self.params.t1 * 1e3,
                    t2=self.params.t2 * 1e3,
                    time=self.params.gate_time_2q,
                    excited_state_population=self.params.thermal_population
                )
            )

            if self.verbose:
                print(f"✓ 热弛豫噪声: T1={self.params.t1}μs, T2={self.params.t2}μs")

        # 2. 额外的门错误（去极化）
        if self.include_gate_errors:
            error_1q_gate = depolarizing_error(self.params.gate_error_1q, 1)
            error_2q_gate = depolarizing_error(self.params.gate_error_2q, 2)

            if self.verbose:
                print(f"✓ 门错误: 单比特={self.params.gate_error_1q:.4f}, "
                      f"双比特={self.params.gate_error_2q:.4f}")

        # 3. 组合噪声
        if self.include_thermal_relaxation and self.include_gate_errors:
            # 组合热弛豫和门错误
            combined_error_1q = error_1q_thermal.compose(error_1q_gate)
            combined_error_2q = error_2q_thermal.compose(error_2q_gate)
        elif self.include_thermal_relaxation:
            combined_error_1q = error_1q_thermal
            combined_error_2q = error_2q_thermal
        elif self.include_gate_errors:
            combined_error_1q = error_1q_gate
            combined_error_2q = error_2q_gate
        else:
            combined_error_1q = None
            combined_error_2q = None

        # 添加组合错误到噪声模型
        if combined_error_1q is not None:
            noise_model.add_all_qubit_quantum_error(
                combined_error_1q,
                ['h', 'rx', 'ry', 'rz', 'x', 'y', 'z']
            )

        if combined_error_2q is not None:
            noise_model.add_all_qubit_quantum_error(
                combined_error_2q,
                ['cx', 'cz', 'rzz']
            )

        # 4. 读取错误
        if self.include_readout_errors:
            p_error = self.params.readout_error
            readout_error_matrix = [
                [1 - p_error, p_error],
                [p_error, 1 - p_error]
            ]
            readout_error = ReadoutError(readout_error_matrix)
            noise_model.add_all_qubit_readout_error(readout_error)

            if self.verbose:
                print(f"✓ 读取错误: {self.params.readout_error:.4f}")

        if self.verbose:
            print("="*60)
            print("噪声模型构建完成")
            print("="*60)

        return noise_model


def create_noise_model(
    noise_type: str = "superconducting",
    **kwargs
) -> NoiseModel:
    """便捷函数：创建噪声模型

    Parameters
    ----------
    noise_type : str, optional
        噪声类型，可选值:
        - 'depolarizing': 去极化噪声
        - 'amplitude_damping': 振幅阻尼噪声
        - 'thermal': 热弛豫噪声
        - 'readout': 读取噪声
        - 'superconducting': 超导综合噪声（默认）
    **kwargs
        传递给NoiseParameters的参数

    Returns
    -------
    NoiseModel
        噪声模型实例

    Examples
    --------
    >>> # 创建超导噪声模型
    >>> noise = create_noise_model('superconducting', t1=50.0, t2=70.0)
    >>>
    >>> # 创建简单的去极化噪声
    >>> noise = create_noise_model('depolarizing', gate_error_1q=0.001)
    """
    params = NoiseParameters(**kwargs)

    noise_models = {
        'depolarizing': DepolarizingNoise,
        'amplitude_damping': AmplitudeDampingNoise,
        'thermal': ThermalRelaxationNoise,
        'readout': ReadoutNoise,
        'superconducting': SuperconductingNoiseModel
    }

    if noise_type not in noise_models:
        raise ValueError(
            f"Unknown noise type: {noise_type}. "
            f"Available types: {list(noise_models.keys())}"
        )

    return noise_models[noise_type](params)


if __name__ == "__main__":
    print("噪声模型模块加载成功!")
    print("\n可用噪声模型:")
    print("  - DepolarizingNoise: 去极化噪声")
    print("  - AmplitudeDampingNoise: 振幅阻尼噪声")
    print("  - ThermalRelaxationNoise: 热弛豫噪声")
    print("  - ReadoutNoise: 读取噪声")
    print("  - SuperconductingNoiseModel: 超导综合噪声")
    print("\n便捷函数:")
    print("  - create_noise_model: 快速创建噪声模型")

    # 测试示例
    print("\n" + "="*60)
    print("示例：创建超导噪声模型")
    print("="*60)
    params = NoiseParameters(
        gate_error_1q=0.001,
        gate_error_2q=0.01,
        t1=50.0,
        t2=70.0,
        readout_error=0.02
    )
    noise = SuperconductingNoiseModel(params, verbose=True)
    qiskit_noise = noise.build()
    print("\n噪声模型摘要:")
    print(noise.summary())
