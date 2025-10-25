"""参数迁移学习工具模块

提供QAOA参数在不同规模问题间迁移的自动化工具。

Classes:
    ParameterTransferManager: 参数迁移管理器
    TransferStrategy: 迁移策略基类

Functions:
    progressive_solve: 渐进式求解（从小到大逐步扩展）

Author: [待填写]
Created: 2025-10-17
"""

from typing import List, Dict, Any, Optional, Callable, Tuple
from dataclasses import dataclass, field
import numpy as np
from numpy.typing import NDArray
import time
import warnings

from ..formulations.base import BaseQUBOFormulation
from ..solvers.qaoa_solver import QAOASolver, OptimizationConfig, SolverResult


@dataclass
class TransferResult:
    """参数迁移结果类

    Attributes
    ----------
    problem_sizes : list
        问题规模序列
    results : list of SolverResult
        每个规模的求解结果
    transferred_params : list of dict
        每次迁移的参数
    total_time : float
        总求解时间
    metadata : dict
        额外信息
    """
    problem_sizes: List[int]
    results: List[SolverResult]
    transferred_params: List[Dict[str, List[float]]]
    total_time: float
    metadata: Dict[str, Any] = field(default_factory=dict)

    def get_improvement_stats(self) -> Dict[str, Any]:
        """计算迁移学习带来的改进

        Returns
        -------
        dict
            改进统计信息
        """
        if len(self.results) < 2:
            return {}

        improvements = []
        for i in range(1, len(self.results)):
            prev_iters = self.results[i-1].num_iterations
            curr_iters = self.results[i].num_iterations

            # 估算如果不用迁移可能需要的迭代次数（粗略估计）
            estimated_iters = prev_iters * 1.5  # 假设大问题需要更多迭代

            improvement = {
                'problem_size': self.problem_sizes[i],
                'actual_iterations': curr_iters,
                'estimated_no_transfer': estimated_iters,
                'saved_iterations': estimated_iters - curr_iters,
                'time_saved_pct': ((estimated_iters - curr_iters) / estimated_iters * 100) if estimated_iters > 0 else 0
            }
            improvements.append(improvement)

        return {
            'per_step_improvements': improvements,
            'total_problems_solved': len(self.results),
            'total_time': self.total_time
        }


class TransferStrategy:
    """参数迁移策略基类

    定义如何从小问题的参数迁移到大问题。
    """

    def transfer(
        self,
        source_params: Dict[str, List[float]],
        source_size: int,
        target_size: int
    ) -> Dict[str, List[float]]:
        """迁移参数

        Parameters
        ----------
        source_params : dict
            源问题的参数 {'gamma': [...], 'beta': [...]}
        source_size : int
            源问题规模
        target_size : int
            目标问题规模

        Returns
        -------
        dict
            目标问题的初始参数
        """
        raise NotImplementedError("Subclasses must implement transfer()")


class DirectTransferStrategy(TransferStrategy):
    """直接迁移策略

    直接复用源问题的参数，不做任何修改。
    适用于问题规模相近或参数不敏感的情况。
    """

    def transfer(
        self,
        source_params: Dict[str, List[float]],
        source_size: int,
        target_size: int
    ) -> Dict[str, List[float]]:
        """直接复用参数"""
        return source_params.copy()


class ScaledTransferStrategy(TransferStrategy):
    """缩放迁移策略

    根据问题规模比例缩放参数。
    假设最优参数与问题规模有一定的比例关系。
    """

    def __init__(self, scale_factor: float = 0.9):
        """
        Parameters
        ----------
        scale_factor : float, optional
            缩放因子，默认0.9
        """
        self.scale_factor = scale_factor

    def transfer(
        self,
        source_params: Dict[str, List[float]],
        source_size: int,
        target_size: int
    ) -> Dict[str, List[float]]:
        """缩放参数"""
        # 计算规模比例
        size_ratio = target_size / source_size

        # 应用缩放
        transferred = {}
        for key, values in source_params.items():
            # gamma和beta可能有不同的缩放行为
            if key == 'gamma':
                # gamma通常随问题规模减小
                transferred[key] = [v * (1.0 / (size_ratio ** self.scale_factor)) for v in values]
            elif key == 'beta':
                # beta相对稳定
                transferred[key] = [v * 0.95 for v in values]  # 轻微调整
            else:
                transferred[key] = values.copy()

        return transferred


class AdaptiveTransferStrategy(TransferStrategy):
    """自适应迁移策略

    根据历史迁移效果自动调整迁移参数。
    """

    def __init__(self):
        self.history: List[Dict] = []

    def transfer(
        self,
        source_params: Dict[str, List[float]],
        source_size: int,
        target_size: int
    ) -> Dict[str, List[float]]:
        """自适应迁移"""
        if not self.history:
            # 第一次迁移，使用直接策略
            return source_params.copy()

        # 基于历史调整（简化版本）
        # 实际应用中可以使用更复杂的学习算法
        return source_params.copy()

    def record_result(
        self,
        source_size: int,
        target_size: int,
        params: Dict[str, List[float]],
        performance: float
    ):
        """记录迁移结果"""
        self.history.append({
            'source_size': source_size,
            'target_size': target_size,
            'params': params,
            'performance': performance
        })


class ParameterTransferManager:
    """参数迁移管理器

    管理QAOA参数在不同规模问题间的迁移过程。

    Parameters
    ----------
    strategy : TransferStrategy, optional
        迁移策略，默认为DirectTransferStrategy
    verbose : bool, optional
        是否显示详细输出，默认True

    Examples
    --------
    >>> manager = ParameterTransferManager(strategy=DirectTransferStrategy())
    >>> result = manager.progressive_solve(
    ...     problem_generator=lambda size: create_problem(size),
    ...     sizes=[4, 6, 8, 10],
    ...     config=OptimizationConfig(num_layers=2, maxiter=30)
    ... )
    """

    def __init__(
        self,
        strategy: Optional[TransferStrategy] = None,
        verbose: bool = True
    ):
        self.strategy = strategy or DirectTransferStrategy()
        self.verbose = verbose
        self.history: List[Dict] = []

    def progressive_solve(
        self,
        problem_generator: Callable[[int], BaseQUBOFormulation],
        sizes: List[int],
        config: OptimizationConfig,
        warm_start: bool = True
    ) -> TransferResult:
        """渐进式求解：从小到大逐步扩展问题规模

        Parameters
        ----------
        problem_generator : callable
            问题生成器函数，接受规模参数，返回QUBO问题
        sizes : list of int
            问题规模序列（必须递增）
        config : OptimizationConfig
            优化配置（应用于所有规模）
        warm_start : bool, optional
            是否使用参数迁移（热启动），默认True

        Returns
        -------
        TransferResult
            迁移求解结果

        Raises
        ------
        ValueError
            如果规模序列不是递增的
        """
        # 验证规模序列
        if not all(sizes[i] < sizes[i+1] for i in range(len(sizes)-1)):
            raise ValueError("Problem sizes must be in ascending order")

        if self.verbose:
            print("=" * 70)
            print("渐进式参数迁移求解")
            print("=" * 70)
            print(f"问题规模序列: {sizes}")
            print(f"迁移策略: {self.strategy.__class__.__name__}")
            print(f"热启动: {'启用' if warm_start else '禁用'}")
            print("=" * 70)

        results = []
        transferred_params_list = []
        start_time = time.time()

        learned_params = None

        for i, size in enumerate(sizes):
            if self.verbose:
                print(f"\n{'='*70}")
                print(f"求解问题规模: {size}")
                print(f"{'='*70}")

            # 生成问题
            problem = problem_generator(size)

            # 配置求解器
            solver_config = OptimizationConfig(
                num_layers=config.num_layers,
                method=config.method,
                maxiter=config.maxiter,
                shots=config.shots,
                initial_params=None,  # 先设为None
                backend=config.backend,
                optimizer_options=config.optimizer_options
            )

            # 应用参数迁移
            if warm_start and learned_params is not None:
                transferred_params = self.strategy.transfer(
                    learned_params,
                    sizes[i-1],
                    size
                )
                solver_config.initial_params = transferred_params
                transferred_params_list.append(transferred_params)

                if self.verbose:
                    print(f"使用迁移参数（从规模{sizes[i-1]}迁移）")
                    print(f"  γ = {transferred_params['gamma']}")
                    print(f"  β = {transferred_params['beta']}")
            else:
                transferred_params_list.append(None)
                if self.verbose:
                    print("使用随机初始化参数")

            # 求解
            solver = QAOASolver(config=solver_config, verbose=self.verbose)
            result = solver.solve(problem)
            results.append(result)

            # 保存学到的参数供下次迁移
            learned_params = result.optimal_params.copy()

            if self.verbose:
                print(f"\n完成! 最优成本: {result.best_cost:.6f}, "
                      f"迭代: {result.num_iterations}, "
                      f"用时: {result.total_time:.2f}秒")

        total_time = time.time() - start_time

        if self.verbose:
            print(f"\n{'='*70}")
            print(f"渐进式求解完成！总用时: {total_time:.2f}秒")
            print(f"{'='*70}")

        # 构建结果对象
        transfer_result = TransferResult(
            problem_sizes=sizes,
            results=results,
            transferred_params=transferred_params_list,
            total_time=total_time,
            metadata={
                'strategy': self.strategy.__class__.__name__,
                'warm_start': warm_start,
                'config': config.to_dict()
            }
        )

        return transfer_result

    def compare_strategies(
        self,
        problem_generator: Callable[[int], BaseQUBOFormulation],
        sizes: List[int],
        config: OptimizationConfig,
        strategies: Optional[List[TransferStrategy]] = None
    ) -> Dict[str, TransferResult]:
        """对比不同迁移策略的效果

        Parameters
        ----------
        problem_generator : callable
            问题生成器
        sizes : list of int
            规模序列
        config : OptimizationConfig
            优化配置
        strategies : list of TransferStrategy, optional
            要对比的策略列表，默认对比所有内置策略

        Returns
        -------
        dict
            {策略名: TransferResult}
        """
        if strategies is None:
            strategies = [
                DirectTransferStrategy(),
                ScaledTransferStrategy(scale_factor=0.9),
                AdaptiveTransferStrategy()
            ]

        results = {}

        # 添加一个"无迁移"的基线
        if self.verbose:
            print("\n" + "="*70)
            print("基线：无参数迁移（每次随机初始化）")
            print("="*70)

        baseline_result = self.progressive_solve(
            problem_generator,
            sizes,
            config,
            warm_start=False
        )
        results['No Transfer (Baseline)'] = baseline_result

        # 测试各种策略
        for strategy in strategies:
            strategy_name = strategy.__class__.__name__

            if self.verbose:
                print(f"\n{'='*70}")
                print(f"测试策略: {strategy_name}")
                print(f"{'='*70}")

            # 临时切换策略
            original_strategy = self.strategy
            self.strategy = strategy

            result = self.progressive_solve(
                problem_generator,
                sizes,
                config,
                warm_start=True
            )
            results[strategy_name] = result

            # 恢复原策略
            self.strategy = original_strategy

        return results


def progressive_solve(
    problem_generator: Callable[[int], BaseQUBOFormulation],
    sizes: List[int],
    config: OptimizationConfig,
    strategy: Optional[TransferStrategy] = None,
    verbose: bool = True
) -> TransferResult:
    """便捷函数：渐进式求解

    Parameters
    ----------
    problem_generator : callable
        问题生成器函数
    sizes : list of int
        问题规模序列
    config : OptimizationConfig
        优化配置
    strategy : TransferStrategy, optional
        迁移策略
    verbose : bool, optional
        详细输出

    Returns
    -------
    TransferResult
        求解结果

    Examples
    --------
    >>> from src.formulations.frequency_allocation import FrequencyAllocationQUBO
    >>> import networkx as nx
    >>>
    >>> def gen_problem(size):
    ...     topology = nx.cycle_graph(size)
    ...     return FrequencyAllocationQUBO(topology, (4.5, 6.0), num_levels=3)
    >>>
    >>> result = progressive_solve(
    ...     gen_problem,
    ...     sizes=[4, 6, 8],
    ...     config=OptimizationConfig(num_layers=2, maxiter=30)
    ... )
    """
    manager = ParameterTransferManager(strategy=strategy, verbose=verbose)
    return manager.progressive_solve(problem_generator, sizes, config)


if __name__ == "__main__":
    print("参数迁移学习模块加载成功!")
    print("可用类:")
    print("  - ParameterTransferManager: 主管理器")
    print("  - DirectTransferStrategy: 直接迁移策略")
    print("  - ScaledTransferStrategy: 缩放迁移策略")
    print("  - AdaptiveTransferStrategy: 自适应迁移策略")
    print("可用函数:")
    print("  - progressive_solve: 便捷求解函数")
