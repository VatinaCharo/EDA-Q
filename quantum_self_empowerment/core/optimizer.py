"""高级参数优化器模块

本模块提供多种优化算法的统一接口，包括经典优化器和量子计算专用优化器。

Classes:
    BaseOptimizer: 优化器基类
    COBYLAOptimizer: COBYLA优化器
    SPSAOptimizer: SPSA（同步扰动随机逼近）优化器
    AdamOptimizer: Adam优化器
    LBFGSBOptimizer: L-BFGS-B优化器
    NelderMeadOptimizer: Nelder-Mead优化器
    OptimizerFactory: 优化器工厂类

Author: [待填写]
Created: 2025-10-17
"""

from typing import Callable, Optional, Dict, Any, Tuple, List
from abc import ABC, abstractmethod
import numpy as np
from numpy.typing import NDArray
import warnings


class OptimizationResult:
    """优化结果类

    Attributes
    ----------
    x : NDArray
        最优参数
    fun : float
        最优目标函数值
    nit : int
        迭代次数
    success : bool
        是否成功收敛
    message : str
        优化器返回信息
    """

    def __init__(
        self,
        x: NDArray,
        fun: float,
        nit: int,
        success: bool = True,
        message: str = ""
    ):
        self.x = x
        self.fun = fun
        self.nit = nit
        self.success = success
        self.message = message


class BaseOptimizer(ABC):
    """优化器基类

    所有优化器都应继承此类并实现minimize方法。

    Parameters
    ----------
    maxiter : int, optional
        最大迭代次数，默认为100
    tol : float, optional
        收敛容忍度，默认为1e-6
    callback : callable, optional
        每次迭代后的回调函数
    """

    def __init__(
        self,
        maxiter: int = 100,
        tol: float = 1e-6,
        callback: Optional[Callable] = None
    ):
        self.maxiter = maxiter
        self.tol = tol
        self.callback = callback
        self.nit = 0

    @abstractmethod
    def minimize(
        self,
        objective: Callable[[NDArray], float],
        x0: NDArray
    ) -> OptimizationResult:
        """最小化目标函数

        Parameters
        ----------
        objective : callable
            目标函数，接受参数数组，返回标量
        x0 : NDArray
            初始参数

        Returns
        -------
        OptimizationResult
            优化结果
        """
        pass

    def _invoke_callback(self, iteration: int, cost: float, params: NDArray) -> None:
        """调用回调函数"""
        if self.callback is not None:
            self.callback(iteration, cost, params)


class COBYLAOptimizer(BaseOptimizer):
    """COBYLA优化器（无约束情况）

    使用scipy.optimize.minimize的COBYLA方法。
    适用于无梯度的约束优化问题。

    Parameters
    ----------
    maxiter : int, optional
        最大迭代次数
    tol : float, optional
        收敛容忍度
    rhobeg : float, optional
        初始信赖域半径，默认为1.0
    callback : callable, optional
        回调函数

    Examples
    --------
    >>> optimizer = COBYLAOptimizer(maxiter=50)
    >>> result = optimizer.minimize(objective_fn, initial_params)
    """

    def __init__(
        self,
        maxiter: int = 100,
        tol: float = 1e-6,
        rhobeg: float = 1.0,
        callback: Optional[Callable] = None
    ):
        super().__init__(maxiter, tol, callback)
        self.rhobeg = rhobeg

    def minimize(
        self,
        objective: Callable[[NDArray], float],
        x0: NDArray
    ) -> OptimizationResult:
        """使用COBYLA方法最小化"""
        from scipy.optimize import minimize

        # 包装目标函数以支持回调
        def wrapped_objective(x):
            cost = objective(x)
            self.nit += 1
            self._invoke_callback(self.nit, cost, x)
            return cost

        result = minimize(
            wrapped_objective,
            x0,
            method='COBYLA',
            options={
                'maxiter': self.maxiter,
                'rhobeg': self.rhobeg,
                'tol': self.tol
            }
        )

        return OptimizationResult(
            x=result.x,
            fun=result.fun,
            nit=self.nit,
            success=result.success,
            message=result.message if hasattr(result, 'message') else ''
        )


class LBFGSBOptimizer(BaseOptimizer):
    """L-BFGS-B优化器

    使用scipy的L-BFGS-B方法，适用于大规模边界约束优化。

    Parameters
    ----------
    maxiter : int, optional
        最大迭代次数
    tol : float, optional
        收敛容忍度
    bounds : list of tuple, optional
        参数边界 [(low, high), ...]
    callback : callable, optional
        回调函数
    """

    def __init__(
        self,
        maxiter: int = 100,
        tol: float = 1e-6,
        bounds: Optional[List[Tuple[float, float]]] = None,
        callback: Optional[Callable] = None
    ):
        super().__init__(maxiter, tol, callback)
        self.bounds = bounds

    def minimize(
        self,
        objective: Callable[[NDArray], float],
        x0: NDArray
    ) -> OptimizationResult:
        """使用L-BFGS-B方法最小化"""
        from scipy.optimize import minimize

        def wrapped_objective(x):
            cost = objective(x)
            self.nit += 1
            self._invoke_callback(self.nit, cost, x)
            return cost

        # 如果没有指定边界，设置为[0, 2π]
        bounds = self.bounds
        if bounds is None:
            bounds = [(0, 2*np.pi)] * len(x0)

        result = minimize(
            wrapped_objective,
            x0,
            method='L-BFGS-B',
            bounds=bounds,
            options={
                'maxiter': self.maxiter,
                'ftol': self.tol
            }
        )

        return OptimizationResult(
            x=result.x,
            fun=result.fun,
            nit=self.nit,
            success=result.success,
            message=result.message if hasattr(result, 'message') else ''
        )


class SPSAOptimizer(BaseOptimizer):
    """SPSA（同步扰动随机逼近）优化器

    SPSA是一种随机优化算法，特别适合量子计算中的参数优化。
    只需要两次函数评估即可估计梯度，对噪声有较好的鲁棒性。

    Parameters
    ----------
    maxiter : int, optional
        最大迭代次数
    a : float, optional
        步长系数，默认为0.16
    c : float, optional
        扰动系数，默认为0.1
    A : float, optional
        稳定性参数，默认为0
    alpha : float, optional
        步长衰减指数，默认为0.602
    gamma : float, optional
        扰动衰减指数，默认为0.101
    callback : callable, optional
        回调函数

    References
    ----------
    .. [1] Spall, J. C. (1998). "Implementation of the simultaneous
           perturbation algorithm for stochastic optimization."
           IEEE Transactions on Aerospace and Electronic Systems.

    Examples
    --------
    >>> optimizer = SPSAOptimizer(maxiter=100, a=0.16, c=0.1)
    >>> result = optimizer.minimize(noisy_objective, initial_params)
    """

    def __init__(
        self,
        maxiter: int = 100,
        a: float = 0.16,
        c: float = 0.1,
        A: float = 0,
        alpha: float = 0.602,
        gamma: float = 0.101,
        callback: Optional[Callable] = None
    ):
        super().__init__(maxiter, 1e-6, callback)
        self.a = a
        self.c = c
        self.A = A
        self.alpha = alpha
        self.gamma = gamma

    def minimize(
        self,
        objective: Callable[[NDArray], float],
        x0: NDArray
    ) -> OptimizationResult:
        """使用SPSA方法最小化"""
        theta = x0.copy()
        best_theta = theta.copy()
        best_cost = objective(theta)

        for k in range(1, self.maxiter + 1):
            # 计算步长和扰动
            a_k = self.a / ((k + self.A) ** self.alpha)
            c_k = self.c / (k ** self.gamma)

            # 生成随机扰动向量（伯努利分布）
            delta = 2 * np.random.randint(0, 2, size=len(theta)) - 1

            # 计算扰动后的目标函数值
            theta_plus = theta + c_k * delta
            theta_minus = theta - c_k * delta

            y_plus = objective(theta_plus)
            y_minus = objective(theta_minus)

            # 估计梯度
            gradient_estimate = (y_plus - y_minus) / (2 * c_k * delta)

            # 更新参数
            theta = theta - a_k * gradient_estimate

            # 约束参数范围到[0, 2π]
            theta = np.mod(theta, 2 * np.pi)

            # 评估当前解
            current_cost = objective(theta)

            # 更新最优解
            if current_cost < best_cost:
                best_cost = current_cost
                best_theta = theta.copy()

            self.nit = k
            self._invoke_callback(k, current_cost, theta)

        return OptimizationResult(
            x=best_theta,
            fun=best_cost,
            nit=self.nit,
            success=True,
            message=f'SPSA completed {self.nit} iterations'
        )


class AdamOptimizer(BaseOptimizer):
    """Adam优化器

    自适应矩估计优化算法，结合了动量和RMSprop的优点。
    常用于深度学习，也适用于量子电路参数优化。

    Parameters
    ----------
    maxiter : int, optional
        最大迭代次数
    lr : float, optional
        学习率，默认为0.01
    beta1 : float, optional
        一阶矩估计的指数衰减率，默认为0.9
    beta2 : float, optional
        二阶矩估计的指数衰减率，默认为0.999
    epsilon : float, optional
        数值稳定性参数，默认为1e-8
    tol : float, optional
        收敛容忍度
    callback : callable, optional
        回调函数

    Examples
    --------
    >>> optimizer = AdamOptimizer(maxiter=100, lr=0.01)
    >>> result = optimizer.minimize(objective_fn, initial_params)
    """

    def __init__(
        self,
        maxiter: int = 100,
        lr: float = 0.01,
        beta1: float = 0.9,
        beta2: float = 0.999,
        epsilon: float = 1e-8,
        tol: float = 1e-6,
        callback: Optional[Callable] = None
    ):
        super().__init__(maxiter, tol, callback)
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon

    def _numerical_gradient(
        self,
        objective: Callable,
        x: NDArray,
        eps: float = 1e-5
    ) -> NDArray:
        """数值梯度计算（中心差分）"""
        grad = np.zeros_like(x)
        for i in range(len(x)):
            x_plus = x.copy()
            x_minus = x.copy()
            x_plus[i] += eps
            x_minus[i] -= eps
            grad[i] = (objective(x_plus) - objective(x_minus)) / (2 * eps)
        return grad

    def minimize(
        self,
        objective: Callable[[NDArray], float],
        x0: NDArray
    ) -> OptimizationResult:
        """使用Adam方法最小化"""
        theta = x0.copy()
        m = np.zeros_like(theta)  # 一阶矩估计
        v = np.zeros_like(theta)  # 二阶矩估计

        best_theta = theta.copy()
        best_cost = objective(theta)

        for t in range(1, self.maxiter + 1):
            # 计算梯度
            grad = self._numerical_gradient(objective, theta)

            # 更新偏差修正的一阶和二阶矩估计
            m = self.beta1 * m + (1 - self.beta1) * grad
            v = self.beta2 * v + (1 - self.beta2) * (grad ** 2)

            # 偏差修正
            m_hat = m / (1 - self.beta1 ** t)
            v_hat = v / (1 - self.beta2 ** t)

            # 更新参数
            theta = theta - self.lr * m_hat / (np.sqrt(v_hat) + self.epsilon)

            # 约束参数范围
            theta = np.mod(theta, 2 * np.pi)

            # 评估当前解
            current_cost = objective(theta)

            # 更新最优解
            if current_cost < best_cost:
                best_cost = current_cost
                best_theta = theta.copy()

            self.nit = t
            self._invoke_callback(t, current_cost, theta)

            # 检查收敛
            if t > 1 and abs(current_cost - best_cost) < self.tol:
                break

        return OptimizationResult(
            x=best_theta,
            fun=best_cost,
            nit=self.nit,
            success=True,
            message=f'Adam completed {self.nit} iterations'
        )


class NelderMeadOptimizer(BaseOptimizer):
    """Nelder-Mead优化器

    单纯形法，不需要梯度信息，适用于非光滑问题。

    Parameters
    ----------
    maxiter : int, optional
        最大迭代次数
    tol : float, optional
        收敛容忍度
    callback : callable, optional
        回调函数
    """

    def __init__(
        self,
        maxiter: int = 100,
        tol: float = 1e-6,
        callback: Optional[Callable] = None
    ):
        super().__init__(maxiter, tol, callback)

    def minimize(
        self,
        objective: Callable[[NDArray], float],
        x0: NDArray
    ) -> OptimizationResult:
        """使用Nelder-Mead方法最小化"""
        from scipy.optimize import minimize

        def wrapped_objective(x):
            cost = objective(x)
            self.nit += 1
            self._invoke_callback(self.nit, cost, x)
            return cost

        result = minimize(
            wrapped_objective,
            x0,
            method='Nelder-Mead',
            options={
                'maxiter': self.maxiter,
                'xatol': self.tol,
                'fatol': self.tol
            }
        )

        return OptimizationResult(
            x=result.x,
            fun=result.fun,
            nit=self.nit,
            success=result.success,
            message=result.message if hasattr(result, 'message') else ''
        )


class OptimizerFactory:
    """优化器工厂类

    提供统一的接口创建不同类型的优化器。

    Examples
    --------
    >>> optimizer = OptimizerFactory.create('SPSA', maxiter=100)
    >>> result = optimizer.minimize(objective_fn, initial_params)
    """

    _optimizers = {
        'COBYLA': COBYLAOptimizer,
        'L-BFGS-B': LBFGSBOptimizer,
        'SPSA': SPSAOptimizer,
        'Adam': AdamOptimizer,
        'Nelder-Mead': NelderMeadOptimizer
    }

    @classmethod
    def create(
        cls,
        method: str,
        **kwargs
    ) -> BaseOptimizer:
        """创建优化器实例

        Parameters
        ----------
        method : str
            优化器名称，可选值：
            - 'COBYLA': COBYLA优化器
            - 'L-BFGS-B': L-BFGS-B优化器
            - 'SPSA': SPSA优化器
            - 'Adam': Adam优化器
            - 'Nelder-Mead': Nelder-Mead优化器
        **kwargs
            优化器特定参数

        Returns
        -------
        BaseOptimizer
            优化器实例

        Raises
        ------
        ValueError
            如果优化器名称无效
        """
        if method not in cls._optimizers:
            available = ', '.join(cls._optimizers.keys())
            raise ValueError(
                f"Unknown optimizer: {method}. "
                f"Available optimizers: {available}"
            )

        optimizer_class = cls._optimizers[method]
        return optimizer_class(**kwargs)

    @classmethod
    def list_optimizers(cls) -> List[str]:
        """列出所有可用的优化器

        Returns
        -------
        list of str
            优化器名称列表
        """
        return list(cls._optimizers.keys())

    @classmethod
    def get_optimizer_info(cls, method: str) -> str:
        """获取优化器信息

        Parameters
        ----------
        method : str
            优化器名称

        Returns
        -------
        str
            优化器描述信息
        """
        descriptions = {
            'COBYLA': '无梯度约束优化，适用于小规模问题',
            'L-BFGS-B': '拟牛顿法，适用于大规模边界约束优化',
            'SPSA': '随机扰动逼近，对噪声鲁棒，适合量子优化',
            'Adam': '自适应矩估计，常用于深度学习和变分算法',
            'Nelder-Mead': '单纯形法，不需要梯度，适用于非光滑问题'
        }
        return descriptions.get(method, '无描述信息')


def compare_optimizers(
    objective: Callable[[NDArray], float],
    x0: NDArray,
    methods: List[str],
    maxiter: int = 100,
    **kwargs
) -> Dict[str, OptimizationResult]:
    """对比多个优化器的性能

    Parameters
    ----------
    objective : callable
        目标函数
    x0 : NDArray
        初始参数
    methods : list of str
        优化器名称列表
    maxiter : int, optional
        最大迭代次数
    **kwargs
        优化器通用参数

    Returns
    -------
    dict
        {优化器名称: 优化结果}

    Examples
    --------
    >>> results = compare_optimizers(
    ...     objective_fn,
    ...     initial_params,
    ...     methods=['COBYLA', 'SPSA', 'Adam'],
    ...     maxiter=50
    ... )
    >>> for name, result in results.items():
    ...     print(f"{name}: cost={result.fun:.4f}")
    """
    results = {}

    for method in methods:
        print(f"\n运行 {method} 优化器...")
        optimizer = OptimizerFactory.create(method, maxiter=maxiter, **kwargs)

        try:
            result = optimizer.minimize(objective, x0.copy())
            results[method] = result
            print(f"  完成！成本: {result.fun:.6f}, 迭代: {result.nit}")
        except Exception as e:
            print(f"  失败: {str(e)}")
            warnings.warn(f"{method} optimizer failed: {str(e)}")

    return results


if __name__ == "__main__":
    # 简单测试
    print("优化器模块测试")
    print("=" * 60)

    # 列出所有优化器
    print("\n可用优化器:")
    for name in OptimizerFactory.list_optimizers():
        info = OptimizerFactory.get_optimizer_info(name)
        print(f"  - {name}: {info}")

    # 测试简单函数
    def test_objective(x):
        """Rosenbrock函数"""
        return sum(100 * (x[1:] - x[:-1]**2)**2 + (1 - x[:-1])**2)

    x0 = np.array([0.0, 0.0, 0.0])

    print("\n\n测试 Rosenbrock 函数优化:")
    print(f"初始点: {x0}")
    print(f"初始成本: {test_objective(x0):.4f}")

    # 测试COBYLA
    print("\n" + "-" * 60)
    optimizer = OptimizerFactory.create('COBYLA', maxiter=50)
    result = optimizer.minimize(test_objective, x0)
    print(f"COBYLA 结果: x={result.x}, f(x)={result.fun:.6f}")

    print("\n优化器模块加载成功!")
