"""QUBO建模工具模块

本模块提供QUBO (Quadratic Unconstrained Binary Optimization) 问题建模的通用工具。

Classes:
    QUBOBuilder: QUBO矩阵构建器
    QUBOProblem: QUBO问题表示

Functions:
    validate_qubo_matrix: 验证QUBO矩阵的有效性
    decode_binary_solution: 解码二进制解

Author: [待填写]
Created: 2025-10-16
Last Modified: 2025-10-16
"""

from typing import Dict, List, Tuple, Optional, Callable, Union
import numpy as np
from numpy.typing import NDArray
import networkx as nx
import warnings


class QUBOProblem:
    """QUBO问题表示类

    封装QUBO问题的所有信息，包括目标函数、约束等。

    Parameters
    ----------
    num_variables : int
        二进制变量数量
    objective_matrix : NDArray[np.float64], optional
        目标函数矩阵Q，默认为零矩阵
    offset : float, optional
        常数偏移量，默认为0

    Attributes
    ----------
    num_variables : int
        变量数量
    Q : NDArray[np.float64]
        QUBO矩阵
    offset : float
        常数偏移
    metadata : dict
        元数据信息

    Examples
    --------
    >>> problem = QUBOProblem(num_variables=4)
    >>> problem.add_quadratic_term(0, 1, 0.5)
    >>> problem.add_linear_term(0, 1.0)
    """

    def __init__(
        self,
        num_variables: int,
        objective_matrix: Optional[NDArray[np.float64]] = None,
        offset: float = 0.0
    ):
        if num_variables <= 0:
            raise ValueError(f"num_variables must be positive, got {num_variables}")

        self.num_variables = num_variables
        self.offset = offset
        self.metadata: Dict = {}

        if objective_matrix is not None:
            if objective_matrix.shape != (num_variables, num_variables):
                raise ValueError(
                    f"objective_matrix shape {objective_matrix.shape} does not match "
                    f"num_variables {num_variables}"
                )
            self.Q = objective_matrix.copy()
        else:
            self.Q = np.zeros((num_variables, num_variables))

    def add_linear_term(self, variable: int, coefficient: float) -> None:
        """添加线性项

        添加形如 c*x_i 的项到目标函数

        Parameters
        ----------
        variable : int
            变量索引
        coefficient : float
            系数

        Raises
        ------
        IndexError
            如果变量索引超出范围
        """
        if variable < 0 or variable >= self.num_variables:
            raise IndexError(
                f"Variable index {variable} out of range [0, {self.num_variables})"
            )

        self.Q[variable, variable] += coefficient

    def add_quadratic_term(
        self,
        variable1: int,
        variable2: int,
        coefficient: float
    ) -> None:
        """添加二次项

        添加形如 c*x_i*x_j 的项到目标函数

        Parameters
        ----------
        variable1 : int
            第一个变量索引
        variable2 : int
            第二个变量索引
        coefficient : float
            系数

        Raises
        ------
        IndexError
            如果变量索引超出范围
        """
        if variable1 < 0 or variable1 >= self.num_variables:
            raise IndexError(f"Variable1 index {variable1} out of range")
        if variable2 < 0 or variable2 >= self.num_variables:
            raise IndexError(f"Variable2 index {variable2} out of range")

        # 将系数添加到上三角矩阵
        if variable1 <= variable2:
            self.Q[variable1, variable2] += coefficient
        else:
            self.Q[variable2, variable1] += coefficient

    def add_constraint_penalty(
        self,
        constraint_func: Callable[[NDArray], float],
        penalty_weight: float,
        variable_indices: Optional[List[int]] = None
    ) -> None:
        """添加约束惩罚项

        将约束转化为惩罚项添加到目标函数

        Parameters
        ----------
        constraint_func : Callable
            约束函数，输入二进制向量，输出惩罚值
        penalty_weight : float
            惩罚权重
        variable_indices : list, optional
            相关变量索引，如果为None则使用所有变量

        Notes
        -----
        这是一个简化版本，实际使用中可能需要更复杂的约束编码
        """
        if variable_indices is None:
            variable_indices = list(range(self.num_variables))

        # 这里只是占位实现，实际应用中需要根据具体约束类型实现
        warnings.warn("add_constraint_penalty is a placeholder implementation")

    def evaluate(self, solution: Union[List[int], NDArray]) -> float:
        """评估解的目标函数值

        Parameters
        ----------
        solution : list or ndarray
            二进制解向量

        Returns
        -------
        float
            目标函数值

        Examples
        --------
        >>> problem = QUBOProblem(2)
        >>> problem.add_linear_term(0, 1.0)
        >>> problem.add_quadratic_term(0, 1, 0.5)
        >>> cost = problem.evaluate([1, 1])
        >>> print(cost)
        1.5
        """
        x = np.array(solution)
        if len(x) != self.num_variables:
            raise ValueError(
                f"Solution length {len(x)} does not match "
                f"num_variables {self.num_variables}"
            )

        return float(np.dot(x, np.dot(self.Q, x))) + self.offset

    def to_dict(self) -> Dict:
        """转换为字典表示

        Returns
        -------
        dict
            包含QUBO问题所有信息的字典
        """
        return {
            'num_variables': self.num_variables,
            'Q': self.Q.tolist(),
            'offset': self.offset,
            'metadata': self.metadata
        }

    def symmetrize(self) -> None:
        """对称化QUBO矩阵

        将上三角矩阵对称化为完整对称矩阵
        """
        self.Q = (self.Q + self.Q.T) / 2


class QUBOBuilder:
    """QUBO矩阵构建器

    提供便捷的方法来构建各种类型的QUBO问题。

    Parameters
    ----------
    penalty_weight : float, optional
        默认惩罚权重，默认为10.0

    Attributes
    ----------
    penalty_weight : float
        惩罚权重
    problems : dict
        存储的QUBO问题

    Examples
    --------
    >>> builder = QUBOBuilder(penalty_weight=10.0)
    >>> G = nx.cycle_graph(4)
    >>> Q = builder.build_graph_coloring(G, num_colors=3)
    """

    def __init__(self, penalty_weight: float = 10.0):
        if penalty_weight <= 0:
            raise ValueError(f"penalty_weight must be positive, got {penalty_weight}")

        self.penalty_weight = penalty_weight
        self.problems: Dict[str, QUBOProblem] = {}

    def build_max_cut(self, graph: nx.Graph) -> NDArray[np.float64]:
        """构建最大割问题的QUBO矩阵

        Max-Cut问题：将图的节点分成两组，使得跨越两组的边数最多

        Parameters
        ----------
        graph : nx.Graph
            无向图

        Returns
        -------
        NDArray[np.float64]
            QUBO矩阵

        Notes
        -----
        目标函数: maximize Σ_{(i,j)∈E} (x_i - x_j)^2
        等价于: minimize Σ_{(i,j)∈E} (1 - 2*x_i - 2*x_j + 4*x_i*x_j)
        """
        n = len(graph.nodes())
        Q = np.zeros((n, n))

        for i, j in graph.edges():
            # 添加线性项
            Q[i, i] -= 2
            Q[j, j] -= 2

            # 添加二次项
            Q[i, j] += 4

        # 添加常数项（在实际优化中可以忽略）
        offset = len(graph.edges())

        problem = QUBOProblem(n, Q, offset)
        self.problems['max_cut'] = problem

        return Q

    def build_graph_coloring(
        self,
        graph: nx.Graph,
        num_colors: int
    ) -> NDArray[np.float64]:
        """构建图着色问题的QUBO矩阵

        Parameters
        ----------
        graph : nx.Graph
            无向图
        num_colors : int
            颜色数量

        Returns
        -------
        NDArray[np.float64]
            QUBO矩阵

        Notes
        -----
        使用one-hot编码：每个节点i有num_colors个二进制变量x_{i,c}
        约束1：每个节点恰好一种颜色 Σ_c x_{i,c} = 1
        约束2：相邻节点不同色 x_{i,c} * x_{j,c} = 0 for (i,j)∈E
        """
        n_nodes = len(graph.nodes())
        n_vars = n_nodes * num_colors
        Q = np.zeros((n_vars, n_vars))

        def var_index(node: int, color: int) -> int:
            """变量索引：节点node的颜色color"""
            return node * num_colors + color

        # 约束1：每个节点恰好一种颜色
        for node in graph.nodes():
            # (Σ_c x_{i,c} - 1)^2 展开
            for c1 in range(num_colors):
                idx1 = var_index(node, c1)
                # 线性项: -2*x_{i,c}
                Q[idx1, idx1] -= 2

                for c2 in range(c1, num_colors):
                    idx2 = var_index(node, c2)
                    # 二次项: 2*x_{i,c1}*x_{i,c2}
                    if c1 == c2:
                        Q[idx1, idx2] += 1  # x^2 = x for binary
                    else:
                        Q[idx1, idx2] += 2

        # 约束2：相邻节点不同色
        for i, j in graph.edges():
            for c in range(num_colors):
                idx_i = var_index(i, c)
                idx_j = var_index(j, c)
                # 惩罚 x_{i,c} * x_{j,c}
                Q[idx_i, idx_j] += self.penalty_weight

        problem = QUBOProblem(n_vars, Q)
        problem.metadata = {
            'problem_type': 'graph_coloring',
            'num_nodes': n_nodes,
            'num_colors': num_colors
        }
        self.problems['graph_coloring'] = problem

        return Q

    def build_from_graph(
        self,
        graph: nx.Graph,
        edge_weights: Optional[Dict[Tuple[int, int], float]] = None
    ) -> NDArray[np.float64]:
        """从图构建通用QUBO矩阵

        Parameters
        ----------
        graph : nx.Graph
            图结构
        edge_weights : dict, optional
            边权重字典，键为(i, j)，值为权重

        Returns
        -------
        NDArray[np.float64]
            QUBO矩阵
        """
        n = len(graph.nodes())
        Q = np.zeros((n, n))

        for i, j in graph.edges():
            weight = 1.0
            if edge_weights is not None:
                weight = edge_weights.get((i, j), edge_weights.get((j, i), 1.0))

            Q[i, j] = weight

        return Q

    def add_equality_constraint(
        self,
        Q: NDArray[np.float64],
        variables: List[int],
        target_value: int,
        penalty: Optional[float] = None
    ) -> NDArray[np.float64]:
        """添加等式约束

        添加约束: Σ_i x_i = target_value

        Parameters
        ----------
        Q : NDArray
            现有QUBO矩阵
        variables : list
            约束涉及的变量索引
        target_value : int
            目标值
        penalty : float, optional
            惩罚权重，默认使用self.penalty_weight

        Returns
        -------
        NDArray[np.float64]
            更新后的QUBO矩阵

        Notes
        -----
        通过添加惩罚项 P * (Σ_i x_i - target)^2 来实现
        """
        if penalty is None:
            penalty = self.penalty_weight

        Q_new = Q.copy()

        # (Σ_i x_i - target)^2 = Σ_i x_i^2 + Σ_{i<j} 2*x_i*x_j - 2*target*Σ_i x_i + target^2
        # 因为 x_i^2 = x_i (二进制)

        # 线性项: (1 - 2*target) * x_i
        for i in variables:
            Q_new[i, i] += penalty * (1 - 2 * target_value)

        # 二次项: 2 * x_i * x_j
        for idx1, i in enumerate(variables):
            for j in variables[idx1 + 1:]:
                Q_new[i, j] += penalty * 2

        # 常数项可以忽略（不影响最优解）

        return Q_new


def validate_qubo_matrix(Q: NDArray[np.float64]) -> bool:
    """验证QUBO矩阵的有效性

    Parameters
    ----------
    Q : NDArray[np.float64]
        QUBO矩阵

    Returns
    -------
    bool
        是否有效

    Raises
    ------
    ValueError
        如果矩阵无效
    """
    if Q.ndim != 2:
        raise ValueError(f"Q must be 2-dimensional, got shape {Q.shape}")

    if Q.shape[0] != Q.shape[1]:
        raise ValueError(f"Q must be square, got shape {Q.shape}")

    if not np.isfinite(Q).all():
        raise ValueError("Q contains non-finite values (NaN or Inf)")

    return True


def decode_binary_solution(
    bitstring: str,
    variable_mapping: Optional[Dict[int, str]] = None
) -> Dict:
    """解码二进制解

    Parameters
    ----------
    bitstring : str
        二进制字符串
    variable_mapping : dict, optional
        变量映射，从索引到变量名

    Returns
    -------
    dict
        解码后的解

    Examples
    --------
    >>> mapping = {0: 'x0', 1: 'x1', 2: 'x2'}
    >>> result = decode_binary_solution('101', mapping)
    >>> print(result)
    {'x0': 1, 'x1': 0, 'x2': 1}
    """
    clean_bits = bitstring.replace(' ', '')
    solution = {}

    for i, bit in enumerate(clean_bits):
        var_name = variable_mapping.get(i, f'x{i}') if variable_mapping else f'x{i}'
        solution[var_name] = int(bit)

    return solution


def qubo_to_ising(Q: NDArray[np.float64]) -> Tuple[NDArray[np.float64], NDArray[np.float64], float]:
    """将QUBO问题转换为Ising模型

    QUBO: min x^T Q x, x ∈ {0,1}^n
    Ising: min s^T J s + h^T s, s ∈ {-1,1}^n

    变换: x_i = (1 + s_i) / 2

    Parameters
    ----------
    Q : NDArray[np.float64]
        QUBO矩阵

    Returns
    -------
    J : NDArray[np.float64]
        Ising耦合矩阵
    h : NDArray[np.float64]
        Ising局部场向量
    offset : float
        常数偏移

    References
    ----------
    .. [1] Lucas, A. "Ising formulations of many NP problems."
           Frontiers in Physics 2 (2014): 5.
    """
    n = Q.shape[0]

    # 对称化Q矩阵
    Q_sym = (Q + Q.T) / 2

    # J矩阵（耦合项）
    J = Q_sym / 4

    # h向量（局部场）
    h = np.sum(Q_sym, axis=1) / 4

    # 常数偏移
    offset = np.sum(Q_sym) / 4

    return J, h, offset


if __name__ == "__main__":
    # 简单测试
    print("QUBO模块加载成功!")
    print("可用类: QUBOProblem, QUBOBuilder")
    print("可用函数: validate_qubo_matrix, decode_binary_solution, qubo_to_ising")

    # 测试示例
    problem = QUBOProblem(num_variables=3)
    problem.add_linear_term(0, 1.0)
    problem.add_quadratic_term(0, 1, 0.5)
    print(f"\n测试QUBO问题:")
    print(f"变量数: {problem.num_variables}")
    print(f"矩阵:\n{problem.Q}")
    print(f"评估[1,1,0]: {problem.evaluate([1, 1, 0])}")
