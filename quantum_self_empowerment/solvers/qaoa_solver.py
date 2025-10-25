"""QAOA求解器高级封装

提供易用的QAOA求解器接口，集成优化、可视化和结果管理功能。

Classes:
    QAOASolver: QAOA求解器主类
    OptimizationConfig: 优化配置类
    SolverResult: 求解结果类

Author: [待填写]
Created: 2025-10-16
"""

from typing import Dict, List, Any, Optional, Callable, Union
from dataclasses import dataclass, field, asdict
import numpy as np
from numpy.typing import NDArray
import json
import time
from datetime import datetime

from ..core.qaoa import QAOAOptimizer
from ..formulations.base import BaseQUBOFormulation


@dataclass
class OptimizationConfig:
    """优化配置类

    Parameters
    ----------
    num_layers : int
        QAOA层数
    method : str
        优化方法 ('COBYLA', 'SPSA', 'NELDER-MEAD', etc.)
    maxiter : int
        最大迭代次数
    initial_params : dict, optional
        初始参数 {'gamma': [...], 'beta': [...]}
    shots : int
        量子电路采样次数
    backend : str
        量子后端
    optimizer_options : dict, optional
        优化器额外选项
    """
    num_layers: int = 1
    method: str = 'COBYLA'
    maxiter: int = 100
    initial_params: Optional[Dict[str, List[float]]] = None
    shots: int = 1024
    backend: str = 'aer_simulator'
    optimizer_options: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return asdict(self)


@dataclass
class SolverResult:
    """求解结果类

    Attributes
    ----------
    best_solution : str
        最优解比特串
    best_cost : float
        最优成本值
    optimal_params : dict
        最优参数
    cost_history : list
        成本历史
    counts : dict
        测量结果计数
    total_time : float
        总求解时间（秒）
    num_iterations : int
        实际迭代次数
    config : OptimizationConfig
        优化配置
    metadata : dict
        额外元数据
    """
    best_solution: str
    best_cost: float
    optimal_params: Dict[str, List[float]]
    cost_history: List[float]
    counts: Dict[str, int]
    total_time: float
    num_iterations: int
    config: OptimizationConfig
    gamma_history: Optional[List[NDArray]] = None
    beta_history: Optional[List[NDArray]] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        result_dict = {
            'best_solution': self.best_solution,
            'best_cost': self.best_cost,
            'optimal_params': self.optimal_params,
            'cost_history': self.cost_history,
            'counts': self.counts,
            'total_time': self.total_time,
            'num_iterations': self.num_iterations,
            'config': self.config.to_dict(),
            'metadata': self.metadata
        }

        # 处理numpy数组
        if self.gamma_history is not None:
            result_dict['gamma_history'] = [g.tolist() for g in self.gamma_history]
        if self.beta_history is not None:
            result_dict['beta_history'] = [b.tolist() for b in self.beta_history]

        return result_dict

    def save(self, filepath: str) -> None:
        """保存结果到JSON文件

        Parameters
        ----------
        filepath : str
            保存路径
        """
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(self.to_dict(), f, indent=2, ensure_ascii=False)

    @classmethod
    def load(cls, filepath: str) -> 'SolverResult':
        """从JSON文件加载结果

        Parameters
        ----------
        filepath : str
            文件路径

        Returns
        -------
        SolverResult
            求解结果对象
        """
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)

        # 重构配置对象
        config = OptimizationConfig(**data['config'])

        # 重构历史数组
        gamma_history = None
        beta_history = None
        if 'gamma_history' in data and data['gamma_history']:
            gamma_history = [np.array(g) for g in data['gamma_history']]
        if 'beta_history' in data and data['beta_history']:
            beta_history = [np.array(b) for b in data['beta_history']]

        return cls(
            best_solution=data['best_solution'],
            best_cost=data['best_cost'],
            optimal_params=data['optimal_params'],
            cost_history=data['cost_history'],
            counts=data['counts'],
            total_time=data['total_time'],
            num_iterations=data['num_iterations'],
            config=config,
            gamma_history=gamma_history,
            beta_history=beta_history,
            metadata=data.get('metadata', {})
        )


class QAOASolver:
    """QAOA求解器高级封装类

    提供统一的接口用于求解QUBO问题，支持多种优化策略、
    进度跟踪、结果可视化等功能。

    Parameters
    ----------
    config : OptimizationConfig, optional
        优化配置
    verbose : bool, optional
        是否显示详细输出
    callback : callable, optional
        每次迭代后的回调函数

    Attributes
    ----------
    config : OptimizationConfig
        当前配置
    optimizer : QAOAOptimizer
        底层QAOA优化器
    verbose : bool
        详细输出开关

    Examples
    --------
    >>> from src.formulations.frequency_allocation import FrequencyAllocationQUBO
    >>> import networkx as nx
    >>>
    >>> # 定义问题
    >>> topology = nx.cycle_graph(4)
    >>> formulation = FrequencyAllocationQUBO(
    ...     topology=topology,
    ...     freq_range=(4.5, 6.0),
    ...     num_levels=4
    ... )
    >>>
    >>> # 配置求解器
    >>> config = OptimizationConfig(num_layers=2, maxiter=50)
    >>> solver = QAOASolver(config=config, verbose=True)
    >>>
    >>> # 求解
    >>> result = solver.solve(formulation)
    >>> print(f"最优成本: {result.best_cost}")
    """

    def __init__(
        self,
        config: Optional[OptimizationConfig] = None,
        verbose: bool = True,
        callback: Optional[Callable] = None
    ):
        self.config = config or OptimizationConfig()
        self.verbose = verbose
        self.user_callback = callback
        self.optimizer: Optional[QAOAOptimizer] = None

        # 内部状态跟踪
        self._iteration_count = 0
        self._gamma_history: List[NDArray] = []
        self._beta_history: List[NDArray] = []
        self._start_time: float = 0.0

    def solve(
        self,
        formulation: Union[BaseQUBOFormulation, NDArray],
        config: Optional[OptimizationConfig] = None
    ) -> SolverResult:
        """求解QUBO问题

        Parameters
        ----------
        formulation : BaseQUBOFormulation or NDArray
            QUBO问题建模对象或QUBO矩阵
        config : OptimizationConfig, optional
            覆盖默认配置

        Returns
        -------
        SolverResult
            求解结果对象

        Raises
        ------
        ValueError
            如果输入参数不合法
        """
        # 使用提供的配置或默认配置
        if config is not None:
            self.config = config

        # 获取QUBO矩阵
        if isinstance(formulation, np.ndarray):
            Q = formulation
            num_qubits = Q.shape[0]
        else:
            Q = formulation.build_qubo()
            num_qubits = Q.shape[0]  # 使用QUBO矩阵的实际大小，而不是num_variables

        # 验证QUBO矩阵
        if Q.shape[0] != Q.shape[1]:
            raise ValueError(f"QUBO矩阵必须是方阵，当前形状: {Q.shape}")

        # 初始化优化器
        self.optimizer = QAOAOptimizer(
            num_qubits=num_qubits,
            num_layers=self.config.num_layers,
            backend=self.config.backend,
            shots=self.config.shots
        )

        # 重置内部状态
        self._reset_tracking()

        if self.verbose:
            print("=" * 60)
            print("QAOA求解器启动")
            print("=" * 60)
            print(f"量子比特数: {num_qubits}")
            print(f"QAOA层数: {self.config.num_layers}")
            print(f"优化方法: {self.config.method}")
            print(f"最大迭代次数: {self.config.maxiter}")
            print(f"采样次数: {self.config.shots}")
            print("-" * 60)

        # 准备回调函数
        def _internal_callback(iteration: int, cost: float, params: NDArray) -> None:
            """内部回调函数"""
            self._iteration_count = iteration

            # 提取gamma和beta
            num_layers = self.config.num_layers
            gamma = params[:num_layers]
            beta = params[num_layers:2*num_layers]

            self._gamma_history.append(gamma.copy())
            self._beta_history.append(beta.copy())

            if self.verbose and iteration % 10 == 0:
                print(f"迭代 {iteration:3d} | 成本: {cost:10.6f} | "
                      f"γ: {gamma} | β: {beta}")

            # 调用用户回调
            if self.user_callback is not None:
                self.user_callback(iteration, cost, params)

        # 开始求解
        self._start_time = time.time()

        # 转换initial_params从dict到numpy数组（如果需要）
        initial_params_array = None
        if self.config.initial_params is not None:
            if isinstance(self.config.initial_params, dict):
                # 从dict格式转换为numpy数组
                gamma = np.array(self.config.initial_params['gamma'])
                beta = np.array(self.config.initial_params['beta'])
                initial_params_array = np.concatenate([gamma, beta])
            else:
                # 假设已经是numpy数组
                initial_params_array = self.config.initial_params

        try:
            result = self.optimizer.optimize(
                Q=Q,
                method=self.config.method,
                maxiter=self.config.maxiter,
                initial_params=initial_params_array,
                callback=_internal_callback
            )

            total_time = time.time() - self._start_time

            # 从QAOAOptimizer的返回值中提取数据
            best_cost = result['optimal_cost']
            best_solution = result['best_solution']
            optimal_params_array = result['optimal_params']
            cost_history = result['history']['costs']
            counts = result['solution_counts']
            num_iterations = result.get('num_iterations', len(cost_history))

            # 将optimal_params转换为字典格式
            num_layers = self.config.num_layers
            optimal_params = {
                'gamma': optimal_params_array[:num_layers].tolist(),
                'beta': optimal_params_array[num_layers:2*num_layers].tolist()
            }

            if self.verbose:
                print("-" * 60)
                print(f"求解完成！总用时: {total_time:.2f}秒")
                print(f"最优成本: {best_cost:.6f}")
                print(f"最优解: {best_solution}")
                print("=" * 60)

            # 构建结果对象
            solver_result = SolverResult(
                best_solution=best_solution,
                best_cost=best_cost,
                optimal_params=optimal_params,
                cost_history=cost_history,
                counts=counts,
                total_time=total_time,
                num_iterations=num_iterations,
                config=self.config,
                gamma_history=self._gamma_history,
                beta_history=self._beta_history,
                metadata={
                    'timestamp': datetime.now().isoformat(),
                    'num_qubits': num_qubits,
                    'qubo_shape': Q.shape,
                    'success': True
                }
            )

            return solver_result

        except Exception as e:
            if self.verbose:
                print(f"求解失败: {str(e)}")
            raise

    def solve_and_visualize(
        self,
        formulation: BaseQUBOFormulation,
        config: Optional[OptimizationConfig] = None,
        save_path: Optional[str] = None
    ) -> SolverResult:
        """求解问题并生成可视化报告

        Parameters
        ----------
        formulation : BaseQUBOFormulation
            QUBO问题建模对象
        config : OptimizationConfig, optional
            优化配置
        save_path : str, optional
            保存图片路径

        Returns
        -------
        SolverResult
            求解结果
        """
        # 求解
        result = self.solve(formulation, config)

        # 生成可视化
        try:
            from ..utils.visualization import create_summary_report
            import matplotlib.pyplot as plt

            # 准备优化结果字典（兼容旧接口）
            opt_result = {
                'best_solution': result.best_solution,
                'best_cost': result.best_cost,
                'cost_history': result.cost_history,
                'counts': result.counts,
                'gamma_history': result.gamma_history,
                'beta_history': result.beta_history
            }

            fig = create_summary_report(opt_result, formulation)

            if save_path:
                fig.savefig(save_path, dpi=150, bbox_inches='tight')
                if self.verbose:
                    print(f"可视化报告已保存至: {save_path}")

            plt.show()

        except ImportError as e:
            if self.verbose:
                print(f"警告: 无法生成可视化 - {str(e)}")

        return result

    def compare_configurations(
        self,
        formulation: BaseQUBOFormulation,
        configs: List[OptimizationConfig],
        config_names: Optional[List[str]] = None
    ) -> Dict[str, SolverResult]:
        """比较多个配置的性能

        Parameters
        ----------
        formulation : BaseQUBOFormulation
            QUBO问题
        configs : list of OptimizationConfig
            配置列表
        config_names : list of str, optional
            配置名称列表

        Returns
        -------
        dict
            {配置名: 求解结果}
        """
        if config_names is None:
            config_names = [f"Config_{i}" for i in range(len(configs))]

        if len(config_names) != len(configs):
            raise ValueError("配置名称数量必须与配置数量相同")

        results = {}

        for name, cfg in zip(config_names, configs):
            if self.verbose:
                print(f"\n{'='*60}")
                print(f"测试配置: {name}")
                print(f"{'='*60}")

            result = self.solve(formulation, config=cfg)
            results[name] = result

        # 打印对比摘要
        if self.verbose:
            self._print_comparison_summary(results)

        return results

    def _print_comparison_summary(
        self,
        results: Dict[str, SolverResult]
    ) -> None:
        """打印对比摘要"""
        print("\n" + "=" * 80)
        print("配置性能对比摘要")
        print("=" * 80)
        print(f"{'配置名':<20} {'最优成本':<15} {'迭代次数':<12} {'用时(秒)':<12}")
        print("-" * 80)

        for name, result in results.items():
            print(f"{name:<20} {result.best_cost:<15.6f} "
                  f"{result.num_iterations:<12} {result.total_time:<12.2f}")

        print("=" * 80)

        # 找出最佳配置
        best_name = min(results.items(), key=lambda x: x[1].best_cost)[0]
        print(f"\n最佳配置: {best_name}")

    def _reset_tracking(self) -> None:
        """重置跟踪状态"""
        self._iteration_count = 0
        self._gamma_history = []
        self._beta_history = []
        self._start_time = 0.0


def quick_solve(
    formulation: BaseQUBOFormulation,
    num_layers: int = 1,
    maxiter: int = 100,
    verbose: bool = True
) -> SolverResult:
    """快速求解函数（便捷接口）

    Parameters
    ----------
    formulation : BaseQUBOFormulation
        QUBO问题
    num_layers : int, optional
        QAOA层数
    maxiter : int, optional
        最大迭代次数
    verbose : bool, optional
        详细输出

    Returns
    -------
    SolverResult
        求解结果

    Examples
    --------
    >>> from src.formulations.frequency_allocation import FrequencyAllocationQUBO
    >>> import networkx as nx
    >>> topology = nx.cycle_graph(4)
    >>> formulation = FrequencyAllocationQUBO(topology, (4.5, 6.0))
    >>> result = quick_solve(formulation, num_layers=2, maxiter=50)
    """
    config = OptimizationConfig(num_layers=num_layers, maxiter=maxiter)
    solver = QAOASolver(config=config, verbose=verbose)
    return solver.solve(formulation)


if __name__ == "__main__":
    print("QAOASolver模块加载成功!")
    print("可用类:")
    print("  - OptimizationConfig: 优化配置")
    print("  - SolverResult: 求解结果")
    print("  - QAOASolver: 主求解器类")
    print("可用函数:")
    print("  - quick_solve: 快速求解函数")
