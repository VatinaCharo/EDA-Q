"""
专业出版级可视化工具

符合高水平中文期刊（如《中国科学》、《物理学报》、《计算机学报》等）的发表标准

特性：
- 高分辨率（300 DPI）
- 专业配色方案
- 中英文双语支持
- 规范的子图标签
- 统一的样式主题
"""

import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib import font_manager
import numpy as np
from typing import Dict, List, Tuple, Optional, Union
import seaborn as sns

# ============================================================================
# 配色方案（来自科研标准配色）
# ============================================================================

# Nature风格配色
NATURE_COLORS = [
    '#E64B35',  # 红色
    '#4DBBD5',  # 青色
    '#00A087',  # 绿色
    '#3C5488',  # 深蓝
    '#F39B7F',  # 橙色
    '#8491B4',  # 紫灰
    '#91D1C2',  # 薄荷绿
    '#DC0000',  # 深红
]

# Science风格配色
SCIENCE_COLORS = [
    '#3B4992',  # 深蓝
    '#EE0000',  # 红色
    '#008B45',  # 绿色
    '#631879',  # 紫色
    '#008280',  # 青色
    '#BB0021',  # 深红
    '#5F559B',  # 紫蓝
    '#A20056',  # 品红
]

# 单色渐变（用于热图）
HEATMAP_CMAP = 'RdYlBu_r'
SEQUENTIAL_CMAP = 'viridis'


# ============================================================================
# 全局样式配置
# ============================================================================

def setup_publication_style(
    style: str = 'nature',
    font_size: int = 10,
    use_latex: bool = False,
    figure_dpi: int = 300
):
    """设置出版级图表样式

    Parameters
    ----------
    style : str
        配色方案，可选 'nature', 'science', 'default'
    font_size : int
        基础字体大小（单位：pt）
    use_latex : bool
        是否使用LaTeX渲染（需要系统安装LaTeX）
    figure_dpi : int
        图像分辨率（DPI）
    """

    # 基础配置
    plt.style.use('seaborn-v0_8-paper')

    # 字体配置（中英文）
    mpl.rcParams['font.sans-serif'] = ['SimHei', 'Arial', 'DejaVu Sans']
    mpl.rcParams['font.family'] = 'sans-serif'
    mpl.rcParams['axes.unicode_minus'] = False

    # LaTeX渲染（可选）
    if use_latex:
        mpl.rcParams['text.usetex'] = True
        mpl.rcParams['text.latex.preamble'] = r'\usepackage{amsmath}'

    # 分辨率设置
    mpl.rcParams['figure.dpi'] = figure_dpi
    mpl.rcParams['savefig.dpi'] = figure_dpi
    mpl.rcParams['savefig.bbox'] = 'tight'
    mpl.rcParams['savefig.pad_inches'] = 0.05

    # 字体大小
    mpl.rcParams['font.size'] = font_size
    mpl.rcParams['axes.labelsize'] = font_size + 1
    mpl.rcParams['axes.titlesize'] = font_size + 2
    mpl.rcParams['xtick.labelsize'] = font_size - 1
    mpl.rcParams['ytick.labelsize'] = font_size - 1
    mpl.rcParams['legend.fontsize'] = font_size - 1

    # 线条和标记
    mpl.rcParams['lines.linewidth'] = 1.5
    mpl.rcParams['lines.markersize'] = 6
    mpl.rcParams['patch.linewidth'] = 1.0

    # 坐标轴
    mpl.rcParams['axes.linewidth'] = 1.0
    mpl.rcParams['axes.grid'] = True
    mpl.rcParams['grid.alpha'] = 0.3
    mpl.rcParams['grid.linestyle'] = '--'
    mpl.rcParams['grid.linewidth'] = 0.5

    # 图例
    mpl.rcParams['legend.frameon'] = True
    mpl.rcParams['legend.framealpha'] = 0.9
    mpl.rcParams['legend.edgecolor'] = '0.8'

    # 配色方案
    if style == 'nature':
        mpl.rcParams['axes.prop_cycle'] = mpl.cycler(color=NATURE_COLORS)
    elif style == 'science':
        mpl.rcParams['axes.prop_cycle'] = mpl.cycler(color=SCIENCE_COLORS)

    print(f"✓ 已设置出版级样式：{style} | 字体大小：{font_size}pt | DPI：{figure_dpi}")


# ============================================================================
# 子图标签工具
# ============================================================================

def add_subfig_label(ax, label: str, x: float = -0.1, y: float = 1.05,
                     fontsize: int = 12, fontweight: str = 'bold'):
    """添加子图标签（a, b, c等）

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        目标坐标轴
    label : str
        标签文本（如 'a', 'b', '(a)', '(b)'）
    x, y : float
        标签位置（相对于坐标轴）
    fontsize : int
        字体大小
    fontweight : str
        字体粗细
    """
    ax.text(x, y, label, transform=ax.transAxes,
            fontsize=fontsize, fontweight=fontweight,
            va='top', ha='right')


def add_subfig_labels(axes, labels: Optional[List[str]] = None,
                     style: str = 'parenthesis', **kwargs):
    """批量添加子图标签

    Parameters
    ----------
    axes : list or np.ndarray
        坐标轴列表或数组
    labels : list of str, optional
        自定义标签，默认为 a, b, c, ...
    style : str
        标签样式：'plain' (a, b), 'parenthesis' ((a), (b))
    """
    if not isinstance(axes, (list, np.ndarray)):
        axes = [axes]

    axes_flat = np.array(axes).flatten()

    if labels is None:
        labels = [chr(97 + i) for i in range(len(axes_flat))]  # a, b, c, ...

    if style == 'parenthesis':
        labels = [f'({label})' for label in labels]

    for ax, label in zip(axes_flat, labels):
        add_subfig_label(ax, label, **kwargs)


# ============================================================================
# 专业绘图函数
# ============================================================================

def plot_convergence_curves(
    data_dict: Dict[str, np.ndarray],
    xlabel: str = "Iteration",
    ylabel: str = "Cost Value",
    title: Optional[str] = None,
    figsize: Tuple[int, int] = (8, 5),
    show_fill: bool = True,
    save_path: Optional[str] = None
) -> plt.Figure:
    """绘制收敛曲线（带误差带）

    Parameters
    ----------
    data_dict : dict
        数据字典，格式为 {算法名: cost历史数组}
        如果值是2D数组，会计算均值和标准差
    xlabel, ylabel : str
        坐标轴标签
    title : str, optional
        图标题
    figsize : tuple
        图像大小
    show_fill : bool
        是否显示误差带
    save_path : str, optional
        保存路径

    Returns
    -------
    fig : matplotlib.figure.Figure
        图像对象
    """
    fig, ax = plt.subplots(figsize=figsize)

    for i, (name, data) in enumerate(data_dict.items()):
        data = np.array(data)

        if data.ndim == 1:
            # 单次运行
            ax.plot(data, label=name, linewidth=2)
        else:
            # 多次运行，绘制均值和标准差
            mean = np.mean(data, axis=0)
            std = np.std(data, axis=0)
            x = np.arange(len(mean))

            ax.plot(x, mean, label=name, linewidth=2)

            if show_fill:
                ax.fill_between(x, mean - std, mean + std, alpha=0.2)

    ax.set_xlabel(xlabel, fontweight='bold')
    ax.set_ylabel(ylabel, fontweight='bold')

    if title:
        ax.set_title(title, fontweight='bold', pad=15)

    ax.legend(loc='best', framealpha=0.9)
    ax.grid(True, alpha=0.3, linestyle='--')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ 图像已保存至: {save_path}")

    return fig


def plot_bar_comparison(
    data_dict: Dict[str, Union[float, List[float]]],
    categories: Optional[List[str]] = None,
    xlabel: str = "",
    ylabel: str = "Value",
    title: Optional[str] = None,
    figsize: Tuple[int, int] = (10, 6),
    show_values: bool = True,
    error_bars: Optional[Dict[str, float]] = None,
    save_path: Optional[str] = None
) -> plt.Figure:
    """绘制柱状图对比

    Parameters
    ----------
    data_dict : dict
        数据字典，格式为 {组名: 值或值列表}
    categories : list of str, optional
        类别名称（x轴标签）
    xlabel, ylabel : str
        坐标轴标签
    title : str, optional
        图标题
    figsize : tuple
        图像大小
    show_values : bool
        是否在柱上显示数值
    error_bars : dict, optional
        误差线数据，格式同data_dict
    save_path : str, optional
        保存路径

    Returns
    -------
    fig : matplotlib.figure.Figure
        图像对象
    """
    fig, ax = plt.subplots(figsize=figsize)

    # 数据准备
    groups = list(data_dict.keys())
    values_list = [data_dict[g] if isinstance(data_dict[g], list) else [data_dict[g]]
                   for g in groups]

    n_groups = len(groups)
    n_bars = len(values_list[0])

    if categories is None:
        categories = [f"类别{i+1}" for i in range(n_bars)]

    # 设置柱的位置
    x = np.arange(n_bars)
    width = 0.8 / n_groups

    # 绘制柱状图
    for i, (group, values) in enumerate(zip(groups, values_list)):
        offset = (i - n_groups/2 + 0.5) * width

        yerr = None
        if error_bars and group in error_bars:
            yerr = error_bars[group] if isinstance(error_bars[group], list) else [error_bars[group]]

        bars = ax.bar(x + offset, values, width, label=group,
                     yerr=yerr, capsize=3, error_kw={'linewidth': 1.5})

        # 在柱上显示数值
        if show_values:
            for bar, val in zip(bars, values):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{val:.2f}',
                       ha='center', va='bottom', fontsize=9)

    ax.set_xlabel(xlabel, fontweight='bold')
    ax.set_ylabel(ylabel, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(categories)

    if title:
        ax.set_title(title, fontweight='bold', pad=15)

    ax.legend(loc='best', framealpha=0.9)
    ax.grid(True, alpha=0.3, axis='y', linestyle='--')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ 图像已保存至: {save_path}")

    return fig


def plot_heatmap(
    matrix: np.ndarray,
    xlabel: str = "",
    ylabel: str = "",
    title: Optional[str] = None,
    cmap: str = HEATMAP_CMAP,
    figsize: Tuple[int, int] = (8, 6),
    annot: bool = True,
    fmt: str = '.2f',
    cbar_label: str = "",
    save_path: Optional[str] = None
) -> plt.Figure:
    """绘制热图

    Parameters
    ----------
    matrix : np.ndarray
        2D矩阵数据
    xlabel, ylabel : str
        坐标轴标签
    title : str, optional
        图标题
    cmap : str
        颜色映射
    figsize : tuple
        图像大小
    annot : bool
        是否在格子中显示数值
    fmt : str
        数值格式
    cbar_label : str
        颜色条标签
    save_path : str, optional
        保存路径

    Returns
    -------
    fig : matplotlib.figure.Figure
        图像对象
    """
    fig, ax = plt.subplots(figsize=figsize)

    im = ax.imshow(matrix, cmap=cmap, aspect='auto')

    # 添加颜色条
    cbar = plt.colorbar(im, ax=ax)
    if cbar_label:
        cbar.set_label(cbar_label, fontweight='bold')

    # 显示数值
    if annot:
        for i in range(matrix.shape[0]):
            for j in range(matrix.shape[1]):
                text = ax.text(j, i, format(matrix[i, j], fmt),
                             ha="center", va="center", color="black", fontsize=9)

    ax.set_xlabel(xlabel, fontweight='bold')
    ax.set_ylabel(ylabel, fontweight='bold')

    if title:
        ax.set_title(title, fontweight='bold', pad=15)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ 图像已保存至: {save_path}")

    return fig


def plot_scatter_with_trend(
    x_data: np.ndarray,
    y_data: np.ndarray,
    xlabel: str = "X",
    ylabel: str = "Y",
    title: Optional[str] = None,
    figsize: Tuple[int, int] = (8, 6),
    show_trend: bool = True,
    trend_order: int = 1,
    point_labels: Optional[List[str]] = None,
    save_path: Optional[str] = None
) -> plt.Figure:
    """绘制散点图（带趋势线）

    Parameters
    ----------
    x_data, y_data : np.ndarray
        数据数组
    xlabel, ylabel : str
        坐标轴标签
    title : str, optional
        图标题
    figsize : tuple
        图像大小
    show_trend : bool
        是否显示趋势线
    trend_order : int
        趋势线阶数（1=线性，2=二次等）
    point_labels : list of str, optional
        数据点标签
    save_path : str, optional
        保存路径

    Returns
    -------
    fig : matplotlib.figure.Figure
        图像对象
    """
    fig, ax = plt.subplots(figsize=figsize)

    # 绘制散点
    ax.scatter(x_data, y_data, s=80, alpha=0.7, edgecolors='black', linewidths=1.5)

    # 添加标签
    if point_labels:
        for i, label in enumerate(point_labels):
            ax.annotate(label, (x_data[i], y_data[i]),
                       xytext=(5, 5), textcoords='offset points',
                       fontsize=9)

    # 趋势线
    if show_trend:
        z = np.polyfit(x_data, y_data, trend_order)
        p = np.poly1d(z)
        x_trend = np.linspace(x_data.min(), x_data.max(), 100)
        ax.plot(x_trend, p(x_trend), 'r--', linewidth=2, alpha=0.7,
               label=f'趋势线 (order={trend_order})')

        # 计算R²
        y_pred = p(x_data)
        ss_res = np.sum((y_data - y_pred) ** 2)
        ss_tot = np.sum((y_data - np.mean(y_data)) ** 2)
        r_squared = 1 - (ss_res / ss_tot)

        ax.text(0.05, 0.95, f'$R^2 = {r_squared:.4f}$',
               transform=ax.transAxes, fontsize=10,
               verticalalignment='top',
               bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    ax.set_xlabel(xlabel, fontweight='bold')
    ax.set_ylabel(ylabel, fontweight='bold')

    if title:
        ax.set_title(title, fontweight='bold', pad=15)

    if show_trend:
        ax.legend(loc='best', framealpha=0.9)

    ax.grid(True, alpha=0.3, linestyle='--')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ 图像已保存至: {save_path}")

    return fig


# ============================================================================
# 量子电路特定可视化
# ============================================================================

def plot_topology_with_frequencies(
    topology,
    frequencies: Dict[int, float],
    title: str = "频率分配拓扑",
    figsize: Tuple[int, int] = (10, 8),
    cmap: str = 'viridis',
    save_path: Optional[str] = None
) -> plt.Figure:
    """绘制带频率分配的拓扑图（专业版）

    Parameters
    ----------
    topology : nx.Graph
        拓扑图
    frequencies : dict
        频率分配方案 {节点ID: 频率值}
    title : str
        图标题
    figsize : tuple
        图像大小
    cmap : str
        颜色映射
    save_path : str, optional
        保存路径

    Returns
    -------
    fig : matplotlib.figure.Figure
        图像对象
    """
    import networkx as nx

    fig, ax = plt.subplots(figsize=figsize)

    # 布局
    pos = nx.spring_layout(topology, seed=42, k=1.5)

    # 节点颜色（频率值）
    node_colors = [frequencies.get(node, 0) for node in topology.nodes()]

    # 绘制边
    nx.draw_networkx_edges(topology, pos, ax=ax, edge_color='gray',
                          width=2, alpha=0.6)

    # 绘制节点
    nodes = nx.draw_networkx_nodes(topology, pos, ax=ax,
                                   node_color=node_colors,
                                   node_size=800,
                                   cmap=cmap,
                                   edgecolors='black',
                                   linewidths=2,
                                   vmin=min(node_colors),
                                   vmax=max(node_colors))

    # 节点标签
    nx.draw_networkx_labels(topology, pos, ax=ax,
                           font_size=12,
                           font_weight='bold',
                           font_color='white')

    # 颜色条
    cbar = plt.colorbar(nodes, ax=ax)
    cbar.set_label('频率 (GHz)', fontweight='bold', fontsize=11)

    ax.set_title(title, fontweight='bold', fontsize=14, pad=20)
    ax.axis('off')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ 图像已保存至: {save_path}")

    return fig


# ============================================================================
# 示例用法
# ============================================================================

if __name__ == "__main__":
    # 设置出版级样式
    setup_publication_style(style='nature', font_size=11, figure_dpi=150)

    # 示例1：收敛曲线
    data = {
        'QAOA': np.random.randn(5, 50).cumsum(axis=1) + np.linspace(10, 5, 50),
        'VQE': np.random.randn(5, 50).cumsum(axis=1) + np.linspace(12, 6, 50),
        'Classical': np.random.randn(5, 50).cumsum(axis=1) + np.linspace(8, 4, 50),
    }

    fig1 = plot_convergence_curves(
        data,
        xlabel="迭代次数 (Iteration)",
        ylabel="成本值 (Cost)",
        title="算法收敛性对比"
    )
    plt.show()

    # 示例2：柱状图对比
    data2 = {
        'QAOA': [15.2, 12.3, 18.5, 14.7],
        'VQE': [16.1, 13.2, 19.3, 15.8],
        '经典算法': [20.5, 18.7, 25.1, 22.3],
    }

    fig2 = plot_bar_comparison(
        data2,
        categories=['4量子比特', '8量子比特', '12量子比特', '16量子比特'],
        xlabel="问题规模",
        ylabel="求解时间 (s)",
        title="不同算法的求解效率对比"
    )
    plt.show()

    print("✓ 可视化工具测试完成")
