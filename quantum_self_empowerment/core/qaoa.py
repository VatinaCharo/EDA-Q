"""QAOA核心实现模块

本模块提供量子近似优化算法(QAOA)的核心实现，包括电路构建、参数优化等功能。

Classes:
    QAOACircuit: QAOA量子电路构建器
    QAOAOptimizer: QAOA参数优化器

Functions:
    create_qaoa_circuit: 便捷函数，创建QAOA电路

Author: [待填写]
Created: 2025-10-16
Last Modified: 2025-10-16
"""

from typing import Tuple, Dict, List, Optional, Callable, Union
import numpy as np
from numpy.typing import NDArray
from qiskit import QuantumCircuit
from qiskit_aer import AerSimulator
from qiskit.primitives import Sampler
import warnings

# 设置警告过滤
warnings.filterwarnings('ignore')


class QAOACircuit:
    """QAOA量子电路构建器

    用于构建不同层数的QAOA电路，支持自定义哈密顿量。

    Parameters
    ----------
    num_qubits : int
        量子比特数量
    num_layers : int, optional
        QAOA层数（p值），默认为1

    Attributes
    ----------
    num_qubits : int
        量子比特数量
    num_layers : int
        QAOA层数
    qc : QuantumCircuit
        量子电路对象

    Examples
    --------
    >>> qaoa = QAOACircuit(num_qubits=4, num_layers=2)
    >>> Q = np.array([[1, 0.5], [0.5, 1]])
    >>> circuit = qaoa.build_circuit(Q, gamma=[0.5, 0.3], beta=[0.2, 0.4])
    >>> print(circuit.depth())

    Notes
    -----
    QAOA电路结构：

    1. 初始化：所有量子比特处于 |+⟩ 态（Hadamard门）
    2. 问题层（重复p次）：
       - Cost Hamiltonian: e^{-iγC}
       - Mixer Hamiltonian: e^{-iβB}
    3. 测量：测量所有量子比特

    References
    ----------
    .. [1] Farhi, E., et al. "A quantum approximate optimization algorithm."
           arXiv:1411.4028 (2014)
    """

    def __init__(self, num_qubits: int, num_layers: int = 1):
        if num_qubits <= 0:
            raise ValueError(f"num_qubits must be positive, got {num_qubits}")
        if num_layers <= 0:
            raise ValueError(f"num_layers must be positive, got {num_layers}")

        self.num_qubits = num_qubits
        self.num_layers = num_layers
        self.qc: Optional[QuantumCircuit] = None

    def build_circuit(
        self,
        Q: NDArray[np.float64],
        gamma: Union[List[float], NDArray[np.float64]],
        beta: Union[List[float], NDArray[np.float64]]
    ) -> QuantumCircuit:
        """构建QAOA电路

        Parameters
        ----------
        Q : NDArray[np.float64]
            QUBO矩阵，形状为 (n, n)
        gamma : list or ndarray
            Cost Hamiltonian参数，长度为 num_layers
        beta : list or ndarray
            Mixer Hamiltonian参数，长度为 num_layers

        Returns
        -------
        QuantumCircuit
            构建好的QAOA量子电路

        Raises
        ------
        ValueError
            如果Q矩阵维度不匹配或参数长度不正确
        """
        # 参数验证
        if Q.shape[0] != self.num_qubits:
            raise ValueError(
                f"QUBO matrix size {Q.shape[0]} does not match "
                f"num_qubits {self.num_qubits}"
            )

        gamma = np.array(gamma)
        beta = np.array(beta)

        if len(gamma) != self.num_layers:
            raise ValueError(
                f"gamma length {len(gamma)} does not match "
                f"num_layers {self.num_layers}"
            )
        if len(beta) != self.num_layers:
            raise ValueError(
                f"beta length {len(beta)} does not match "
                f"num_layers {self.num_layers}"
            )

        # 创建量子电路（不预先创建经典比特，让measure_all自动添加）
        qc = QuantumCircuit(self.num_qubits)

        # 初始化：均匀叠加态
        qc.h(range(self.num_qubits))

        # QAOA层
        for layer in range(self.num_layers):
            # Cost Hamiltonian
            self._apply_cost_hamiltonian(qc, Q, gamma[layer])

            # Mixer Hamiltonian
            self._apply_mixer_hamiltonian(qc, beta[layer])

        # 测量
        qc.measure_all()

        self.qc = qc
        return qc

    def _apply_cost_hamiltonian(
        self,
        qc: QuantumCircuit,
        Q: NDArray[np.float64],
        gamma: float
    ) -> None:
        """应用Cost Hamiltonian (问题哈密顿量)

        实现 e^{-iγC}，其中C是QUBO问题对应的算符

        Parameters
        ----------
        qc : QuantumCircuit
            量子电路
        Q : NDArray[np.float64]
            QUBO矩阵
        gamma : float
            参数γ
        """
        # 双量子比特项 (Q[i,j] * Z_i * Z_j)
        for i in range(self.num_qubits):
            for j in range(i + 1, self.num_qubits):
                if abs(Q[i, j]) > 1e-10:  # 忽略接近零的项
                    qc.rzz(2 * gamma * Q[i, j], i, j)

        # 单量子比特项 (Q[i,i] * Z_i)
        for i in range(self.num_qubits):
            if abs(Q[i, i]) > 1e-10:
                qc.rz(2 * gamma * Q[i, i], i)

    def _apply_mixer_hamiltonian(
        self,
        qc: QuantumCircuit,
        beta: float
    ) -> None:
        """应用Mixer Hamiltonian (混合哈密顿量)

        实现 e^{-iβB}，其中B = Σ X_i

        Parameters
        ----------
        qc : QuantumCircuit
            量子电路
        beta : float
            参数β
        """
        # X旋转
        for i in range(self.num_qubits):
            qc.rx(2 * beta, i)

    def get_circuit_depth(self) -> int:
        """获取电路深度

        Returns
        -------
        int
            电路深度
        """
        if self.qc is None:
            raise RuntimeError("Circuit not built yet. Call build_circuit() first.")
        return self.qc.depth()

    def get_num_parameters(self) -> int:
        """获取参数数量

        Returns
        -------
        int
            参数总数 (2 * num_layers)
        """
        return 2 * self.num_layers


class QAOAOptimizer:
    """QAOA参数优化器

    用于优化QAOA参数以最小化目标函数。

    Parameters
    ----------
    num_qubits : int
        量子比特数量
    num_layers : int, optional
        QAOA层数，默认为1
    backend : str, optional
        量子后端，默认为 'aer_simulator'
    shots : int, optional
        测量次数，默认为1024

    Attributes
    ----------
    circuit_builder : QAOACircuit
        电路构建器
    backend : AerSimulator
        量子后端
    shots : int
        测量次数
    optimization_history : dict
        优化历史记录

    Examples
    --------
    >>> import numpy as np
    >>> Q = np.array([[1, 0.5], [0.5, 1]])
    >>> optimizer = QAOAOptimizer(num_qubits=2, num_layers=1)
    >>> result = optimizer.optimize(Q, method='COBYLA', maxiter=100)
    >>> print(f"Optimal cost: {result['optimal_cost']:.4f}")
    """

    def __init__(
        self,
        num_qubits: int,
        num_layers: int = 1,
        backend: str = 'aer_simulator',
        shots: int = 1024
    ):
        self.circuit_builder = QAOACircuit(num_qubits, num_layers)
        # 使用automatic方法让Qiskit自动选择合适的模拟器
        # 对于小规模问题使用statevector，大规模问题使用matrix_product_state
        # 注意：stabilizer方法不支持rzz等非Clifford门，因此不适用于QAOA
        if num_qubits <= 12:
            # 小规模（≤12量子比特）：使用statevector（最快最准确）
            self.backend = AerSimulator(method='statevector')
        elif num_qubits <= 20:
            # 中等规模（13-20量子比特）：使用matrix_product_state（近似但内存友好）
            self.backend = AerSimulator(method='matrix_product_state')
        else:
            # 大规模（>20量子比特）：使用automatic让Qiskit自动选择
            self.backend = AerSimulator(method='automatic')
        self.shots = shots
        self.optimization_history: Dict[str, List] = {
            'costs': [],
            'params': [],
            'iteration': 0
        }

    def compute_expectation(
        self,
        Q: NDArray[np.float64],
        params: NDArray[np.float64]
    ) -> float:
        """计算期望值

        Parameters
        ----------
        Q : NDArray[np.float64]
            QUBO矩阵
        params : NDArray[np.float64]
            QAOA参数 [γ_1, ..., γ_p, β_1, ..., β_p]

        Returns
        -------
        float
            目标函数期望值
        """
        num_layers = self.circuit_builder.num_layers
        gamma = params[:num_layers]
        beta = params[num_layers:]

        try:
            # 构建电路
            qc = self.circuit_builder.build_circuit(Q, gamma, beta)

            # 运行电路
            job = self.backend.run(qc, shots=self.shots)
            result = job.result()
            counts = result.get_counts()

            if not counts:
                return 1e6  # 返回大的惩罚值

            # 计算期望值
            expectation = 0.0
            total_shots = sum(counts.values())

            for bitstring, count in counts.items():
                # 清理比特串（移除空格）
                clean_bits = bitstring.replace(' ', '')
                if len(clean_bits) != self.circuit_builder.num_qubits:
                    continue

                # 转换为二进制向量
                x = np.array([int(bit) for bit in clean_bits])

                # 计算成本
                cost = np.dot(x, np.dot(Q, x))

                # 累加期望值
                probability = count / total_shots
                expectation += probability * cost

            return expectation

        except Exception as e:
            warnings.warn(f"Error in compute_expectation: {e}")
            return 1e6

    def optimize(
        self,
        Q: NDArray[np.float64],
        method: str = 'COBYLA',
        maxiter: int = 100,
        initial_params: Optional[NDArray[np.float64]] = None,
        callback: Optional[Callable] = None,
        **optimizer_kwargs
    ) -> Dict:
        """优化QAOA参数

        Parameters
        ----------
        Q : NDArray[np.float64]
            QUBO矩阵
        method : str, optional
            优化方法，可选值:
            - 'COBYLA': COBYLA优化器（默认）
            - 'L-BFGS-B': L-BFGS-B优化器
            - 'SPSA': SPSA优化器（适合量子噪声环境）
            - 'Adam': Adam优化器
            - 'Nelder-Mead': Nelder-Mead单纯形法
        maxiter : int, optional
            最大迭代次数，默认为100
        initial_params : NDArray, optional
            初始参数，如果为None则随机初始化
        callback : Callable, optional
            每次迭代后的回调函数
        **optimizer_kwargs
            优化器特定参数

        Returns
        -------
        dict
            优化结果，包含:
            - 'optimal_params': 最优参数
            - 'optimal_cost': 最优成本
            - 'history': 优化历史
            - 'best_solution': 最优解的比特串
            - 'solution_counts': 解的概率分布
        """
        from .optimizer import OptimizerFactory

        # 重置优化历史
        self.optimization_history = {'costs': [], 'params': [], 'iteration': 0}

        # 初始化参数
        if initial_params is None:
            num_params = self.circuit_builder.get_num_parameters()
            initial_params = np.random.uniform(0, np.pi, num_params)

        # 定义目标函数
        def objective(params):
            cost = self.compute_expectation(Q, params)

            # 记录历史
            self.optimization_history['costs'].append(cost)
            self.optimization_history['params'].append(params.copy())
            self.optimization_history['iteration'] += 1

            # 调用回调
            if callback is not None:
                callback(self.optimization_history['iteration'], cost, params)

            return cost

        # 创建优化器并执行优化
        try:
            optimizer = OptimizerFactory.create(
                method,
                maxiter=maxiter,
                callback=callback,
                **optimizer_kwargs
            )
            result = optimizer.minimize(objective, initial_params)
        except ValueError:
            # 如果不是OptimizerFactory支持的方法，回退到scipy.optimize
            from scipy.optimize import minimize
            warnings.warn(
                f"Method '{method}' not found in OptimizerFactory, "
                f"falling back to scipy.optimize.minimize"
            )
            result_scipy = minimize(
                objective,
                initial_params,
                method=method,
                options={'maxiter': maxiter}
            )
            # 包装成OptimizationResult格式
            from .optimizer import OptimizationResult
            result = OptimizationResult(
                x=result_scipy.x,
                fun=result_scipy.fun,
                nit=self.optimization_history['iteration'],
                success=result_scipy.success if hasattr(result_scipy, 'success') else True
            )

        # 获取最优解的分布
        optimal_gamma = result.x[:self.circuit_builder.num_layers]
        optimal_beta = result.x[self.circuit_builder.num_layers:]
        qc = self.circuit_builder.build_circuit(Q, optimal_gamma, optimal_beta)

        job = self.backend.run(qc, shots=self.shots)
        counts = job.result().get_counts()

        # 找到最优解
        best_solution = max(counts.keys(), key=lambda k: counts[k])

        return {
            'optimal_params': result.x,
            'optimal_cost': result.fun,
            'history': self.optimization_history,
            'best_solution': best_solution.replace(' ', ''),
            'solution_counts': counts,
            'num_iterations': result.nit
        }


def create_qaoa_circuit(
    num_qubits: int,
    Q: NDArray[np.float64],
    gamma: Union[float, List[float]],
    beta: Union[float, List[float]],
    num_layers: int = 1
) -> QuantumCircuit:
    """便捷函数：创建QAOA电路

    Parameters
    ----------
    num_qubits : int
        量子比特数量
    Q : NDArray[np.float64]
        QUBO矩阵
    gamma : float or list
        γ参数
    beta : float or list
        β参数
    num_layers : int, optional
        层数，默认为1

    Returns
    -------
    QuantumCircuit
        QAOA量子电路

    Examples
    --------
    >>> import numpy as np
    >>> Q = np.array([[1, 0.5], [0.5, 1]])
    >>> qc = create_qaoa_circuit(2, Q, gamma=0.5, beta=0.3)
    """
    if isinstance(gamma, (int, float)):
        gamma = [gamma] * num_layers
    if isinstance(beta, (int, float)):
        beta = [beta] * num_layers

    qaoa = QAOACircuit(num_qubits, num_layers)
    return qaoa.build_circuit(Q, gamma, beta)


if __name__ == "__main__":
    # 简单测试
    import doctest
    doctest.testmod()

    print("QAOA模块加载成功!")
    print(f"可用类: QAOACircuit, QAOAOptimizer")
    print(f"可用函数: create_qaoa_circuit")
