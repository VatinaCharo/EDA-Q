"""可视化工具模块

提供QAOA、QUBO问题和频率分配的可视化功能。

Functions:
    plot_qubo_matrix: 绘制QUBO矩阵热图
    plot_optimization_history: 绘制优化历史曲线
    plot_parameter_evolution: 绘制参数演化过程
    plot_solution_distribution: 绘制解空间分布
    plot_topology_with_frequencies: 绘制带频率标注的拓扑图
    plot_constraint_violations: 绘制约束违反情况
    plot_performance_comparison: 绘制性能对比图
    create_summary_report: 创建综合分析报告

Author: [待填写]
Created: 2025-10-16
"""

from typing import Dict, List, Any, Optional, Tuple, Union
import numpy as np
from numpy.typing import NDArray
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
import networkx as nx
import platform

# 配置中文字体支持
def configure_chinese_fonts():
    """配置matplotlib中文字体"""
    system = platform.system()

    if system == 'Windows':
        # Windows系统常用中文字体
        fonts = ['Microsoft YaHei', 'SimHei', 'KaiTi', 'SimSun']
    elif system == 'Darwin':  # macOS
        fonts = ['PingFang SC', 'Heiti SC', 'STHeiti', 'Arial Unicode MS']
    else:  # Linux
        fonts = ['WenQuanYi Micro Hei', 'Noto Sans CJK SC', 'Droid Sans Fallback']

    # 尝试设置字体
    for font in fonts:
        try:
            plt.rcParams['font.sans-serif'] = [font] + plt.rcParams['font.sans-serif']
            break
        except:
            continue

    # 解决负号显示问题
    plt.rcParams['axes.unicode_minus'] = False

# 自动配置中文字体
configure_chinese_fonts()


def set_plot_style(style: str = 'default') -> None:
    """设置绘图风格

    Parameters
    ----------
    style : str, optional
        风格名称，可选 'default', 'paper', 'presentation'
    """
    if style == 'paper':
        plt.rcParams.update({
            'font.size': 10,
            'axes.labelsize': 11,
            'axes.titlesize': 12,
            'xtick.labelsize': 9,
            'ytick.labelsize': 9,
            'legend.fontsize': 9,
            'figure.titlesize': 13,
            'figure.dpi': 100
        })
    elif style == 'presentation':
        plt.rcParams.update({
            'font.size': 14,
            'axes.labelsize': 16,
            'axes.titlesize': 18,
            'xtick.labelsize': 13,
            'ytick.labelsize': 13,
            'legend.fontsize': 13,
            'figure.titlesize': 20,
            'figure.dpi': 100
        })
    else:
        plt.rcParams.update(plt.rcParamsDefault)


def plot_qubo_matrix(
    Q: NDArray[np.float64],
    ax: Optional[plt.Axes] = None,
    title: str = 'QUBO矩阵',
    cmap: str = 'RdBu_r',
    show_values: bool = True
) -> plt.Axes:
    """绘制QUBO矩阵热图

    Parameters
    ----------
    Q : NDArray
        QUBO矩阵
    ax : plt.Axes, optional
        绘图坐标轴
    title : str, optional
        图标题
    cmap : str, optional
        颜色映射
    show_values : bool, optional
        是否显示数值

    Returns
    -------
    plt.Axes
        绘图坐标轴

    Examples
    --------
    >>> Q = np.array([[1, 2], [2, -1]])
    >>> ax = plot_qubo_matrix(Q)
    >>> plt.show()
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 6))

    # 绘制热图
    im = ax.imshow(Q, cmap=cmap, aspect='auto', interpolation='nearest')

    # 添加颜色条
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('权重', rotation=270, labelpad=20)

    # 设置坐标轴
    n = Q.shape[0]
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels([f'x{i}' for i in range(n)])
    ax.set_yticklabels([f'x{i}' for i in range(n)])

    # 显示数值
    if show_values:
        for i in range(n):
            for j in range(n):
                value = Q[i, j]
                if abs(value) > 1e-10:
                    color = 'white' if abs(value) > abs(Q).max() * 0.5 else 'black'
                    ax.text(j, i, f'{value:.2f}',
                           ha='center', va='center', color=color, fontsize=8)

    ax.set_title(title)
    ax.set_xlabel('变量索引')
    ax.set_ylabel('变量索引')

    return ax


def plot_optimization_history(
    cost_history: List[float],
    ax: Optional[plt.Axes] = None,
    title: str = 'QAOA优化历史',
    show_best: bool = True,
    color: str = 'steelblue'
) -> plt.Axes:
    """绘制优化历史曲线

    Parameters
    ----------
    cost_history : list
        成本函数历史
    ax : plt.Axes, optional
        绘图坐标轴
    title : str, optional
        图标题
    show_best : bool, optional
        是否标注最佳点
    color : str, optional
        曲线颜色

    Returns
    -------
    plt.Axes
        绘图坐标轴
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 6))

    iterations = range(len(cost_history))
    ax.plot(iterations, cost_history, '-o', color=color,
            linewidth=2, markersize=4, alpha=0.7, label='成本值')

    # 标注最佳点
    if show_best:
        best_idx = np.argmin(cost_history)
        best_cost = cost_history[best_idx]
        ax.plot(best_idx, best_cost, 'r*', markersize=15,
               label=f'最优解 (iter={best_idx}, cost={best_cost:.4f})')
        ax.axhline(y=best_cost, color='r', linestyle='--', alpha=0.3)

    ax.set_xlabel('迭代次数')
    ax.set_ylabel('成本函数值')
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)

    return ax


def plot_parameter_evolution(
    gamma_history: List[NDArray],
    beta_history: List[NDArray],
    ax: Optional[plt.Axes] = None,
    title: str = 'QAOA参数演化'
) -> plt.Axes:
    """绘制QAOA参数演化过程

    Parameters
    ----------
    gamma_history : list of NDArray
        gamma参数历史
    beta_history : list of NDArray
        beta参数历史
    ax : plt.Axes, optional
        绘图坐标轴
    title : str, optional
        图标题

    Returns
    -------
    plt.Axes
        绘图坐标轴
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 6))

    # 转换为数组
    gamma_arr = np.array(gamma_history)
    beta_arr = np.array(beta_history)

    iterations = range(len(gamma_history))
    num_layers = gamma_arr.shape[1]

    # 绘制gamma参数
    for layer in range(num_layers):
        ax.plot(iterations, gamma_arr[:, layer],
               label=f'γ_{layer+1}', marker='o', markersize=3)

    # 绘制beta参数
    for layer in range(num_layers):
        ax.plot(iterations, beta_arr[:, layer],
               label=f'β_{layer+1}', marker='s', markersize=3, linestyle='--')

    ax.set_xlabel('迭代次数')
    ax.set_ylabel('参数值')
    ax.set_title(title)
    ax.legend(ncol=2, fontsize=8)
    ax.grid(True, alpha=0.3)

    return ax


def plot_solution_distribution(
    counts: Dict[str, int],
    ax: Optional[plt.Axes] = None,
    title: str = '解空间分布',
    top_k: int = 10,
    highlight_best: Optional[str] = None
) -> plt.Axes:
    """绘制解空间分布柱状图

    Parameters
    ----------
    counts : dict
        解的计数字典
    ax : plt.Axes, optional
        绘图坐标轴
    title : str, optional
        图标题
    top_k : int, optional
        显示前k个解
    highlight_best : str, optional
        高亮显示的最佳解

    Returns
    -------
    plt.Axes
        绘图坐标轴
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(12, 6))

    # 按计数排序
    sorted_items = sorted(counts.items(), key=lambda x: x[1], reverse=True)[:top_k]
    solutions = [item[0] for item in sorted_items]
    frequencies = [item[1] for item in sorted_items]

    # 计算概率
    total = sum(counts.values())
    probabilities = [f / total for f in frequencies]

    # 设置颜色
    colors = ['crimson' if sol == highlight_best else 'steelblue'
             for sol in solutions]

    # 绘制柱状图
    bars = ax.bar(range(len(solutions)), probabilities, color=colors, alpha=0.7)

    # 添加数值标签
    for i, (bar, prob) in enumerate(zip(bars, probabilities)):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
               f'{prob:.3f}', ha='center', va='bottom', fontsize=8)

    ax.set_xlabel('解 (比特串)')
    ax.set_ylabel('概率')
    ax.set_title(f'{title} (显示前{top_k}个)')
    ax.set_xticks(range(len(solutions)))
    ax.set_xticklabels(solutions, rotation=45, ha='right')
    ax.grid(True, axis='y', alpha=0.3)

    # 添加图例
    if highlight_best:
        best_patch = mpatches.Patch(color='crimson', label='最优解')
        other_patch = mpatches.Patch(color='steelblue', label='其他解')
        ax.legend(handles=[best_patch, other_patch])

    return ax


def plot_topology_with_frequencies(
    topology: nx.Graph,
    frequencies: Dict[int, float],
    ax: Optional[plt.Axes] = None,
    title: str = '频率分配拓扑图',
    min_separation: float = 0.2,
    layout: str = 'spring',
    node_size: int = 800,
    cmap: str = 'viridis'
) -> plt.Axes:
    """绘制带频率标注的量子芯片拓扑图

    Parameters
    ----------
    topology : nx.Graph
        量子芯片拓扑
    frequencies : dict
        频率分配字典 {qubit_id: frequency}
    ax : plt.Axes, optional
        绘图坐标轴
    title : str, optional
        图标题
    min_separation : float, optional
        最小频率间隔（用于判断约束违反）
    layout : str, optional
        布局类型 ('spring', 'circular', 'kamada_kawai')
    node_size : int, optional
        节点大小
    cmap : str, optional
        颜色映射

    Returns
    -------
    plt.Axes
        绘图坐标轴
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 8))

    # 选择布局
    if layout == 'spring':
        pos = nx.spring_layout(topology, seed=42)
    elif layout == 'circular':
        pos = nx.circular_layout(topology)
    elif layout == 'kamada_kawai':
        pos = nx.kamada_kawai_layout(topology)
    else:
        pos = nx.spring_layout(topology)

    # 节点颜色基于频率
    node_colors = [frequencies.get(node, 0) for node in topology.nodes()]

    # 绘制节点
    nodes = nx.draw_networkx_nodes(
        topology, pos,
        node_color=node_colors,
        cmap=cmap,
        node_size=node_size,
        ax=ax,
        vmin=min(node_colors),
        vmax=max(node_colors)
    )

    # 添加颜色条
    plt.colorbar(nodes, ax=ax, label='频率 (GHz)')

    # 绘制边（根据是否违反约束着色）
    edge_colors = []
    edge_widths = []
    for i, j in topology.edges():
        freq_diff = abs(frequencies.get(i, 0) - frequencies.get(j, 0))
        if freq_diff < min_separation:
            edge_colors.append('red')
            edge_widths.append(3)
        else:
            edge_colors.append('gray')
            edge_widths.append(2)

    nx.draw_networkx_edges(
        topology, pos,
        edge_color=edge_colors,
        width=edge_widths,
        ax=ax,
        alpha=0.6
    )

    # 添加标签（显示频率）
    labels = {
        node: f"Q{node}\n{frequencies.get(node, 0):.2f} GHz"
        for node in topology.nodes()
    }
    nx.draw_networkx_labels(
        topology, pos,
        labels,
        font_size=9,
        font_weight='bold',
        ax=ax
    )

    # 添加图例
    red_patch = mpatches.Patch(color='red', label='违反约束')
    gray_patch = mpatches.Patch(color='gray', label='满足约束')
    ax.legend(handles=[red_patch, gray_patch], loc='upper right')

    ax.set_title(title)
    ax.axis('off')

    return ax


def plot_constraint_violations(
    topology: nx.Graph,
    frequencies: Dict[int, float],
    min_separation: float,
    ax: Optional[plt.Axes] = None,
    title: str = '约束违反统计'
) -> plt.Axes:
    """绘制约束违反统计图

    Parameters
    ----------
    topology : nx.Graph
        量子芯片拓扑
    frequencies : dict
        频率分配
    min_separation : float
        最小频率间隔
    ax : plt.Axes, optional
        绘图坐标轴
    title : str, optional
        图标题

    Returns
    -------
    plt.Axes
        绘图坐标轴
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 6))

    # 计算每条边的频率间隔
    edge_labels = []
    separations = []
    violation_status = []

    for i, j in topology.edges():
        freq_diff = abs(frequencies.get(i, 0) - frequencies.get(j, 0))
        edge_labels.append(f"({i},{j})")
        separations.append(freq_diff)
        violation_status.append(freq_diff < min_separation)

    # 设置颜色
    colors = ['red' if v else 'green' for v in violation_status]

    # 绘制柱状图
    bars = ax.bar(range(len(edge_labels)), separations, color=colors, alpha=0.7)

    # 添加最小间隔线
    ax.axhline(y=min_separation, color='orange', linestyle='--',
              linewidth=2, label=f'最小间隔 = {min_separation} GHz')

    # 添加数值标签
    for i, (bar, sep) in enumerate(zip(bars, separations)):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
               f'{sep:.3f}', ha='center', va='bottom', fontsize=8)

    ax.set_xlabel('量子比特对')
    ax.set_ylabel('频率间隔 (GHz)')
    ax.set_title(title)
    ax.set_xticks(range(len(edge_labels)))
    ax.set_xticklabels(edge_labels, rotation=45, ha='right')
    ax.grid(True, axis='y', alpha=0.3)

    # 统计信息
    num_violations = sum(violation_status)
    total_constraints = len(violation_status)
    satisfaction_rate = (total_constraints - num_violations) / total_constraints

    # 添加图例和统计
    red_patch = mpatches.Patch(color='red', label=f'违反约束 ({num_violations})')
    green_patch = mpatches.Patch(color='green',
                                 label=f'满足约束 ({total_constraints - num_violations})')
    ax.legend(handles=[red_patch, green_patch,
                      plt.Line2D([0], [0], color='orange', linestyle='--',
                                label=f'满足率: {satisfaction_rate:.1%}')],
             loc='upper right')

    return ax


def plot_performance_comparison(
    metrics_dict: Dict[str, Dict[str, float]],
    ax: Optional[plt.Axes] = None,
    title: str = '性能对比',
    metrics_to_plot: Optional[List[str]] = None
) -> plt.Axes:
    """绘制多个方案的性能对比图

    Parameters
    ----------
    metrics_dict : dict
        性能指标字典 {方案名: {指标名: 值}}
    ax : plt.Axes, optional
        绘图坐标轴
    title : str, optional
        图标题
    metrics_to_plot : list of str, optional
        要绘制的指标列表

    Returns
    -------
    plt.Axes
        绘图坐标轴

    Examples
    --------
    >>> metrics = {
    ...     'QAOA-1': {'approximation_ratio': 0.85, 'time': 10.5},
    ...     'QAOA-2': {'approximation_ratio': 0.90, 'time': 15.2}
    ... }
    >>> ax = plot_performance_comparison(metrics)
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 6))

    # 确定要绘制的指标
    if metrics_to_plot is None:
        all_metrics = set()
        for m in metrics_dict.values():
            all_metrics.update(m.keys())
        metrics_to_plot = list(all_metrics)

    # 准备数据
    methods = list(metrics_dict.keys())
    x = np.arange(len(methods))
    width = 0.8 / len(metrics_to_plot)

    # 绘制分组柱状图
    for i, metric in enumerate(metrics_to_plot):
        values = [metrics_dict[method].get(metric, 0) for method in methods]
        offset = (i - len(metrics_to_plot)/2 + 0.5) * width
        ax.bar(x + offset, values, width, label=metric, alpha=0.8)

    ax.set_xlabel('方法')
    ax.set_ylabel('指标值')
    ax.set_title(title)
    ax.set_xticks(x)
    ax.set_xticklabels(methods, rotation=45, ha='right')
    ax.legend()
    ax.grid(True, axis='y', alpha=0.3)

    return ax


def create_summary_report(
    optimization_result: Dict[str, Any],
    formulation: Any,
    figsize: Tuple[int, int] = (16, 12)
) -> plt.Figure:
    """创建QAOA求解综合分析报告

    Parameters
    ----------
    optimization_result : dict
        QAOA优化结果
    formulation : FrequencyAllocationQUBO
        问题建模对象
    figsize : tuple, optional
        图像大小

    Returns
    -------
    plt.Figure
        完整的分析报告图

    Notes
    -----
    创建包含以下内容的4x2子图布局:
    - QUBO矩阵热图
    - 优化历史曲线
    - 参数演化图
    - 解空间分布
    - 拓扑与频率分配
    - 约束违反统计
    - 性能指标雷达图
    - 收敛性分析
    """
    fig = plt.figure(figsize=figsize)
    gs = GridSpec(4, 2, figure=fig, hspace=0.3, wspace=0.3)

    # 1. QUBO矩阵
    ax1 = fig.add_subplot(gs[0, 0])
    plot_qubo_matrix(formulation.qubo_matrix, ax=ax1, show_values=False)

    # 2. 优化历史
    ax2 = fig.add_subplot(gs[0, 1])
    plot_optimization_history(optimization_result.get('cost_history', []), ax=ax2)

    # 3. 参数演化
    ax3 = fig.add_subplot(gs[1, 0])
    if 'gamma_history' in optimization_result and 'beta_history' in optimization_result:
        plot_parameter_evolution(
            optimization_result['gamma_history'],
            optimization_result['beta_history'],
            ax=ax3
        )

    # 4. 解空间分布
    ax4 = fig.add_subplot(gs[1, 1])
    plot_solution_distribution(
        optimization_result.get('counts', {}),
        ax=ax4,
        highlight_best=optimization_result.get('best_solution')
    )

    # 5. 拓扑与频率分配
    ax5 = fig.add_subplot(gs[2, :])
    solution = formulation.decode_solution(optimization_result['best_solution'])
    plot_topology_with_frequencies(
        formulation.topology,
        solution['frequencies'],
        ax=ax5,
        min_separation=formulation.min_separation
    )

    # 6. 约束违反统计
    ax6 = fig.add_subplot(gs[3, 0])
    plot_constraint_violations(
        formulation.topology,
        solution['frequencies'],
        formulation.min_separation,
        ax=ax6
    )

    # 7. 性能摘要（文本）
    ax7 = fig.add_subplot(gs[3, 1])
    ax7.axis('off')

    # 准备摘要文本
    summary_text = "=== QAOA求解摘要 ===\n\n"
    summary_text += f"最优成本: {optimization_result.get('best_cost', 'N/A'):.4f}\n"
    summary_text += f"总迭代次数: {len(optimization_result.get('cost_history', []))}\n"
    summary_text += f"最优解: {optimization_result.get('best_solution', 'N/A')}\n"
    summary_text += f"约束满足: {'是' if solution.get('is_valid') else '否'}\n"
    summary_text += f"违反数量: {solution.get('num_violations', 'N/A')}\n\n"

    stats = formulation.get_frequency_statistics(solution)
    summary_text += "=== 频率统计 ===\n\n"
    summary_text += f"平均频率: {stats.get('mean_frequency', 0):.3f} GHz\n"
    summary_text += f"频率标准差: {stats.get('std_frequency', 0):.3f} GHz\n"
    summary_text += f"唯一频率数: {stats.get('num_unique_frequencies', 0)}\n"
    summary_text += f"最小频率差: {stats.get('min_freq_difference', 0):.3f} GHz\n"
    summary_text += f"平均频率差: {stats.get('mean_freq_difference', 0):.3f} GHz\n"
    summary_text += f"约束满足率: {stats.get('constraint_satisfaction_rate', 0):.1%}\n"

    ax7.text(0.1, 0.95, summary_text, transform=ax7.transAxes,
            fontsize=10, verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))

    fig.suptitle('QAOA频率分配优化 - 综合分析报告', fontsize=16, fontweight='bold')

    return fig


if __name__ == "__main__":
    print("Visualization模块加载成功!")
    print("可用函数:")
    print("  - set_plot_style")
    print("  - plot_qubo_matrix")
    print("  - plot_optimization_history")
    print("  - plot_parameter_evolution")
    print("  - plot_solution_distribution")
    print("  - plot_topology_with_frequencies")
    print("  - plot_constraint_violations")
    print("  - plot_performance_comparison")
    print("  - create_summary_report")
