"""自适应惩罚权重模块

提供QUBO问题中惩罚权重的自适应调整机制，以提高求解质量。

Classes:
    AdaptivePenaltyOptimizer: 自适应惩罚权重优化器
    PenaltyAdjustmentStrategy: 惩罚调整策略基类

Functions:
    estimate_optimal_penalty: 估计最优惩罚权重

Author: [待填写]
Created: 2025-10-17
"""

from typing import Dict, Any, Optional, Callable, List, Tuple
from abc import ABC, abstractmethod
import numpy as np
from numpy.typing import NDArray
import warnings


class PenaltyAdjustmentStrategy(ABC):
    """惩罚权重调整策略基类"""

    @abstractmethod
    def adjust(
        self,
        current_penalty: float,
        violation_rate: float,
        iteration: int
    ) -> float:
        """调整惩罚权重

        Parameters
        ----------
        current_penalty : float
            当前惩罚权重
        violation_rate : float
            约束违反率 (0.0 - 1.0)
        iteration : int
            当前迭代次数

        Returns
        -------
        float
            调整后的惩罚权重
        """
        pass


class MultiplicativeStrategy(PenaltyAdjustmentStrategy):
    """乘法调整策略

    当违反率高时增加惩罚，违反率低时减小惩罚。
    """

    def __init__(
        self,
        increase_factor: float = 1.5,
        decrease_factor: float = 0.9,
        target_violation_rate: float = 0.1
    ):
        """
        Parameters
        ----------
        increase_factor : float, optional
            惩罚增加因子，默认1.5
        decrease_factor : float, optional
            惩罚减小因子，默认0.9
        target_violation_rate : float, optional
            目标违反率，默认0.1
        """
        self.increase_factor = increase_factor
        self.decrease_factor = decrease_factor
        self.target_violation_rate = target_violation_rate

    def adjust(
        self,
        current_penalty: float,
        violation_rate: float,
        iteration: int
    ) -> float:
        """乘法调整"""
        if violation_rate > self.target_violation_rate:
            # 违反率太高，增加惩罚
            return current_penalty * self.increase_factor
        elif violation_rate < self.target_violation_rate / 2:
            # 违反率很低，可以减小惩罚
            return current_penalty * self.decrease_factor
        else:
            # 在合理范围内，保持不变
            return current_penalty


class AdditiveStrategy(PenaltyAdjustmentStrategy):
    """加法调整策略

    根据违反率线性调整惩罚。
    """

    def __init__(
        self,
        adjustment_rate: float = 2.0,
        target_violation_rate: float = 0.1
    ):
        """
        Parameters
        ----------
        adjustment_rate : float, optional
            调整速率，默认2.0
        target_violation_rate : float, optional
            目标违反率，默认0.1
        """
        self.adjustment_rate = adjustment_rate
        self.target_violation_rate = target_violation_rate

    def adjust(
        self,
        current_penalty: float,
        violation_rate: float,
        iteration: int
    ) -> float:
        """加法调整"""
        error = violation_rate - self.target_violation_rate
        adjustment = self.adjustment_rate * error
        new_penalty = current_penalty + adjustment

        # 确保惩罚权重为正
        return max(0.1, new_penalty)


class AdaptiveStrategy(PenaltyAdjustmentStrategy):
    """自适应调整策略

    基于历史违反率趋势动态调整。
    """

    def __init__(
        self,
        window_size: int = 5,
        aggressive: bool = False
    ):
        """
        Parameters
        ----------
        window_size : int, optional
            历史窗口大小，默认5
        aggressive : bool, optional
            是否采用激进调整，默认False
        """
        self.window_size = window_size
        self.aggressive = aggressive
        self.history: List[float] = []

    def adjust(
        self,
        current_penalty: float,
        violation_rate: float,
        iteration: int
    ) -> float:
        """自适应调整"""
        self.history.append(violation_rate)

        # 保持窗口大小
        if len(self.history) > self.window_size:
            self.history.pop(0)

        if len(self.history) < 2:
            # 历史不足，使用简单规则
            if violation_rate > 0.2:
                return current_penalty * 1.3
            return current_penalty

        # 计算趋势
        recent_avg = np.mean(self.history[-3:]) if len(self.history) >= 3 else violation_rate
        trend = recent_avg - np.mean(self.history[:-1])

        # 根据趋势调整
        if trend > 0.05:  # 违反率上升
            factor = 1.5 if self.aggressive else 1.3
            return current_penalty * factor
        elif trend < -0.05:  # 违反率下降
            factor = 0.85 if self.aggressive else 0.9
            return current_penalty * factor
        else:
            # 稳定，微调
            if recent_avg > 0.15:
                return current_penalty * 1.1
            elif recent_avg < 0.05:
                return current_penalty * 0.95
            return current_penalty


class AdaptivePenaltyOptimizer:
    """自适应惩罚权重优化器

    在求解过程中动态调整QUBO惩罚权重。

    Parameters
    ----------
    initial_penalty : float
        初始惩罚权重
    strategy : PenaltyAdjustmentStrategy, optional
        调整策略，默认为MultiplicativeStrategy
    min_penalty : float, optional
        最小惩罚权重，默认0.1
    max_penalty : float, optional
        最大惩罚权重，默认1000.0
    patience : int, optional
        在调整前等待的迭代次数，默认3

    Examples
    --------
    >>> optimizer = AdaptivePenaltyOptimizer(
    ...     initial_penalty=10.0,
    ...     strategy=MultiplicativeStrategy()
    ... )
    >>> # 在每次迭代后更新
    >>> new_penalty = optimizer.update(violation_rate=0.25)
    """

    def __init__(
        self,
        initial_penalty: float,
        strategy: Optional[PenaltyAdjustmentStrategy] = None,
        min_penalty: float = 0.1,
        max_penalty: float = 1000.0,
        patience: int = 3
    ):
        self.current_penalty = initial_penalty
        self.strategy = strategy or MultiplicativeStrategy()
        self.min_penalty = min_penalty
        self.max_penalty = max_penalty
        self.patience = patience

        self.iteration = 0
        self.history: List[Dict[str, Any]] = []
        self.best_penalty = initial_penalty
        self.best_violation_rate = float('inf')

    def update(
        self,
        violation_rate: float,
        force_update: bool = False
    ) -> float:
        """更新惩罚权重

        Parameters
        ----------
        violation_rate : float
            当前约束违反率 (0.0 - 1.0)
        force_update : bool, optional
            是否强制更新，忽略patience，默认False

        Returns
        -------
        float
            更新后的惩罚权重
        """
        self.iteration += 1

        # 记录历史
        self.history.append({
            'iteration': self.iteration,
            'penalty': self.current_penalty,
            'violation_rate': violation_rate
        })

        # 更新最佳值
        if violation_rate < self.best_violation_rate:
            self.best_violation_rate = violation_rate
            self.best_penalty = self.current_penalty

        # 检查是否需要调整
        if not force_update and self.iteration % self.patience != 0:
            return self.current_penalty

        # 应用调整策略
        new_penalty = self.strategy.adjust(
            self.current_penalty,
            violation_rate,
            self.iteration
        )

        # 限制范围
        new_penalty = np.clip(new_penalty, self.min_penalty, self.max_penalty)

        # 更新
        old_penalty = self.current_penalty
        self.current_penalty = new_penalty

        return self.current_penalty

    def get_penalty(self) -> float:
        """获取当前惩罚权重

        Returns
        -------
        float
            当前惩罚权重
        """
        return self.current_penalty

    def get_best_penalty(self) -> Tuple[float, float]:
        """获取历史最佳惩罚权重及对应的违反率

        Returns
        -------
        tuple of (float, float)
            (最佳惩罚权重, 最低违反率)
        """
        return self.best_penalty, self.best_violation_rate

    def get_history(self) -> List[Dict[str, Any]]:
        """获取调整历史

        Returns
        -------
        list of dict
            历史记录
        """
        return self.history

    def reset(self, initial_penalty: Optional[float] = None):
        """重置优化器

        Parameters
        ----------
        initial_penalty : float, optional
            新的初始惩罚权重，如果为None则使用当前值
        """
        if initial_penalty is not None:
            self.current_penalty = initial_penalty
        self.iteration = 0
        self.history = []
        self.best_violation_rate = float('inf')


def estimate_optimal_penalty(
    problem_matrix: NDArray[np.float64],
    constraint_indices: List[Tuple[int, int]],
    num_samples: int = 10
) -> float:
    """估计最优惩罚权重

    通过分析QUBO矩阵的数值范围估计合适的惩罚权重。

    Parameters
    ----------
    problem_matrix : NDArray
        问题部分的QUBO矩阵（不含惩罚项）
    constraint_indices : list of tuple
        约束项在矩阵中的索引列表
    num_samples : int, optional
        采样次数，默认10

    Returns
    -------
    float
        估计的惩罚权重

    Notes
    -----
    估计原则：惩罚权重应该与目标函数值在同一数量级，
    以确保约束得到足够重视但不会完全主导优化。

    Examples
    --------
    >>> Q_problem = np.array([[1, 0.5], [0.5, 1]])
    >>> constraint_idx = [(0, 1)]
    >>> penalty = estimate_optimal_penalty(Q_problem, constraint_idx)
    """
    if problem_matrix.size == 0:
        return 10.0  # 默认值

    # 计算目标函数的典型规模
    # 方法1：矩阵元素的平均绝对值
    avg_magnitude = np.mean(np.abs(problem_matrix[problem_matrix != 0]))

    # 方法2：随机采样一些比特串，估算目标函数值范围
    n = problem_matrix.shape[0]
    objective_values = []

    for _ in range(num_samples):
        # 随机生成比特串
        x = np.random.randint(0, 2, n)
        # 计算目标函数值
        obj_value = np.dot(x, np.dot(problem_matrix, x))
        objective_values.append(obj_value)

    if objective_values:
        avg_objective = np.mean(np.abs(objective_values))
        std_objective = np.std(objective_values)

        # 惩罚权重应该足够大以影响优化，但不能太大
        # 通常设置为目标函数平均值的倍数
        estimated_penalty = max(avg_magnitude, avg_objective) * 5.0

        # 考虑方差，如果目标函数变化大，需要更大的惩罚
        if std_objective > 0:
            estimated_penalty = max(estimated_penalty, std_objective * 3.0)
    else:
        estimated_penalty = avg_magnitude * 10.0

    # 确保在合理范围内
    estimated_penalty = np.clip(estimated_penalty, 1.0, 100.0)

    return float(estimated_penalty)


def find_optimal_penalty_grid_search(
    formulation,
    solver_func: Callable,
    penalty_range: Tuple[float, float] = (1.0, 100.0),
    num_trials: int = 10,
    metric: str = 'violation_weighted'
) -> Tuple[float, Dict[str, Any]]:
    """网格搜索寻找最优惩罚权重

    Parameters
    ----------
    formulation : BaseQUBOFormulation
        QUBO问题建模对象（需要有penalty_weight属性）
    solver_func : callable
        求解器函数，接受formulation，返回结果
    penalty_range : tuple of float, optional
        搜索范围 (min, max)
    num_trials : int, optional
        尝试的惩罚值数量
    metric : str, optional
        评估指标，可选: 'violation_weighted', 'cost', 'violation_rate'

    Returns
    -------
    tuple of (float, dict)
        (最优惩罚权重, 详细结果字典)

    Examples
    --------
    >>> def my_solver(form):
    ...     solver = QAOASolver(config=OptimizationConfig(num_layers=1, maxiter=20))
    ...     return solver.solve(form)
    >>>
    >>> best_penalty, details = find_optimal_penalty_grid_search(
    ...     formulation,
    ...     my_solver,
    ...     penalty_range=(5.0, 50.0),
    ...     num_trials=5
    ... )
    """
    # 生成惩罚权重候选值（对数空间）
    penalties = np.logspace(
        np.log10(penalty_range[0]),
        np.log10(penalty_range[1]),
        num_trials
    )

    results = []

    print(f"网格搜索最优惩罚权重 (范围: {penalty_range}, 试验: {num_trials})")
    print("=" * 70)

    for penalty in penalties:
        print(f"\n测试惩罚权重: {penalty:.2f}")

        # 设置惩罚权重
        original_penalty = formulation.penalty_weight
        formulation.penalty_weight = penalty

        # 重新构建QUBO（如果需要）
        if hasattr(formulation, 'qubo_matrix'):
            formulation.qubo_matrix = None  # 清除缓存

        try:
            # 求解
            result = solver_func(formulation)

            # 解码并评估
            solution = formulation.decode_solution(result.best_solution)

            violation_rate = solution.get('num_violations', 0)
            if hasattr(formulation, 'num_qubits'):
                violation_rate = violation_rate / max(1, formulation.num_qubits)

            # 计算综合评分
            if metric == 'violation_weighted':
                # 成本和违反率加权组合
                score = result.best_cost + penalty * violation_rate * 10
            elif metric == 'cost':
                score = result.best_cost
            elif metric == 'violation_rate':
                score = violation_rate
            else:
                raise ValueError(f"Unknown metric: {metric}")

            results.append({
                'penalty': penalty,
                'cost': result.best_cost,
                'violation_rate': violation_rate,
                'is_valid': solution.get('is_valid', False),
                'score': score
            })

            print(f"  成本: {result.best_cost:.6f}, "
                  f"违反率: {violation_rate:.3f}, "
                  f"分数: {score:.6f}")

        except Exception as e:
            print(f"  失败: {str(e)}")
            results.append({
                'penalty': penalty,
                'cost': float('inf'),
                'violation_rate': 1.0,
                'is_valid': False,
                'score': float('inf')
            })

        # 恢复原始惩罚权重
        formulation.penalty_weight = original_penalty

    # 找出最佳惩罚权重
    valid_results = [r for r in results if r['score'] < float('inf')]

    if not valid_results:
        warnings.warn("所有惩罚权重都失败了，返回默认值")
        return penalty_range[0], {'results': results}

    best_result = min(valid_results, key=lambda x: x['score'])
    best_penalty = best_result['penalty']

    print(f"\n{'='*70}")
    print(f"最优惩罚权重: {best_penalty:.2f}")
    print(f"  成本: {best_result['cost']:.6f}")
    print(f"  违反率: {best_result['violation_rate']:.3f}")
    print(f"  约束满足: {'✓' if best_result['is_valid'] else '✗'}")
    print(f"{'='*70}")

    return best_penalty, {'results': results, 'best': best_result}


if __name__ == "__main__":
    print("自适应惩罚权重模块加载成功!")
    print("可用类:")
    print("  - AdaptivePenaltyOptimizer: 主优化器")
    print("  - MultiplicativeStrategy: 乘法调整策略")
    print("  - AdditiveStrategy: 加法调整策略")
    print("  - AdaptiveStrategy: 自适应调整策略")
    print("可用函数:")
    print("  - estimate_optimal_penalty: 估计最优惩罚")
    print("  - find_optimal_penalty_grid_search: 网格搜索最优惩罚")
