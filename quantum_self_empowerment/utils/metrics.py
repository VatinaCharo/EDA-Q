"""评估指标计算模块

提供QAOA和QUBO问题求解的各类评估指标计算功能。

Functions:
    compute_approximation_ratio: 计算近似比
    compute_constraint_satisfaction_rate: 计算约束满足率
    compute_solution_diversity: 计算解的多样性

Author: [待填写]
Created: 2025-10-16
"""

from typing import Dict, List, Any, Optional
import numpy as np
from numpy.typing import NDArray


def compute_approximation_ratio(
    solution_cost: float,
    optimal_cost: float,
    worst_cost: Optional[float] = None
) -> float:
    """计算QAOA近似比

    Parameters
    ----------
    solution_cost : float
        QAOA求得的解的成本
    optimal_cost : float
        已知最优解的成本
    worst_cost : float, optional
        最差解的成本，如果提供则计算归一化的近似比

    Returns
    -------
    float
        近似比，值越接近1表示解越好

    Examples
    --------
    >>> ratio = compute_approximation_ratio(15.0, 10.0, 20.0)
    >>> print(f"近似比: {ratio:.2f}")
    近似比: 0.50
    """
    if worst_cost is not None:
        # 归一化近似比: (worst - solution) / (worst - optimal)
        if worst_cost == optimal_cost:
            return 1.0 if solution_cost == optimal_cost else 0.0

        ratio = (worst_cost - solution_cost) / (worst_cost - optimal_cost)
        return max(0.0, min(1.0, ratio))
    else:
        # 简单近似比: optimal / solution
        if solution_cost == 0:
            return 1.0 if optimal_cost == 0 else 0.0

        ratio = optimal_cost / solution_cost
        return min(ratio, 1.0)  # 限制在[0, 1]


def compute_constraint_satisfaction_rate(
    num_violations: int,
    total_constraints: int
) -> float:
    """计算约束满足率

    Parameters
    ----------
    num_violations : int
        违反的约束数量
    total_constraints : int
        总约束数量

    Returns
    -------
    float
        约束满足率，范围[0, 1]

    Examples
    --------
    >>> csr = compute_constraint_satisfaction_rate(2, 10)
    >>> print(f"约束满足率: {csr:.1%}")
    约束满足率: 80.0%
    """
    if total_constraints == 0:
        return 1.0

    return (total_constraints - num_violations) / total_constraints


def compute_solution_diversity(
    solution_counts: Dict[str, int]
) -> Dict[str, float]:
    """计算解空间的多样性指标

    Parameters
    ----------
    solution_counts : dict
        解的计数字典，键为比特串，值为出现次数

    Returns
    -------
    dict
        多样性指标，包括:
        - 'num_unique_solutions': 唯一解的数量
        - 'entropy': 解分布的熵
        - 'top1_probability': 最优解的概率
        - 'top5_probability': 前5个解的总概率
        - 'effective_solutions': 有效解数量（概率>1%）

    Examples
    --------
    >>> counts = {'00': 500, '01': 300, '10': 150, '11': 50}
    >>> diversity = compute_solution_diversity(counts)
    >>> print(f"解空间熵: {diversity['entropy']:.3f}")
    """
    if not solution_counts:
        return {
            'num_unique_solutions': 0,
            'entropy': 0.0,
            'top1_probability': 0.0,
            'top5_probability': 0.0,
            'effective_solutions': 0
        }

    total_shots = sum(solution_counts.values())
    probabilities = np.array([count / total_shots for count in solution_counts.values()])

    # 计算熵
    entropy = -np.sum(probabilities * np.log2(probabilities + 1e-10))

    # 按概率排序
    sorted_probs = sorted(probabilities, reverse=True)

    # Top解的概率
    top1_prob = sorted_probs[0] if len(sorted_probs) > 0 else 0.0
    top5_prob = sum(sorted_probs[:5])

    # 有效解数量（概率>1%）
    effective_sols = sum(1 for p in probabilities if p > 0.01)

    return {
        'num_unique_solutions': len(solution_counts),
        'entropy': float(entropy),
        'top1_probability': float(top1_prob),
        'top5_probability': float(top5_prob),
        'effective_solutions': effective_sols
    }


def compute_convergence_metrics(
    cost_history: List[float]
) -> Dict[str, Any]:
    """计算优化收敛性指标

    Parameters
    ----------
    cost_history : list
        成本函数历史记录

    Returns
    -------
    dict
        收敛性指标
    """
    if not cost_history or len(cost_history) < 2:
        return {
            'total_iterations': len(cost_history),
            'improvement': 0.0,
            'convergence_rate': 0.0,
            'is_converged': False
        }

    costs = np.array(cost_history)

    # 总改进
    improvement = costs[0] - costs[-1]
    improvement_ratio = improvement / abs(costs[0]) if costs[0] != 0 else 0.0

    # 收敛速度（后10次迭代的变化）
    if len(costs) >= 10:
        recent_change = np.std(costs[-10:])
        convergence_rate = recent_change / np.mean(np.abs(costs[-10:]))
    else:
        convergence_rate = np.std(costs) / (np.mean(np.abs(costs)) + 1e-10)

    # 判断是否收敛（最近10次迭代变化<1%）
    is_converged = convergence_rate < 0.01

    return {
        'total_iterations': len(costs),
        'improvement': float(improvement),
        'improvement_ratio': float(improvement_ratio),
        'convergence_rate': float(convergence_rate),
        'is_converged': bool(is_converged),
        'best_cost': float(np.min(costs)),
        'final_cost': float(costs[-1])
    }


if __name__ == "__main__":
    print("Metrics模块加载成功!")
    print("可用函数:")
    print("  - compute_approximation_ratio")
    print("  - compute_constraint_satisfaction_rate")
    print("  - compute_solution_diversity")
    print("  - compute_convergence_metrics")
