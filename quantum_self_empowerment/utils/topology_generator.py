"""超导量子芯片拓扑生成器

生成符合实际超导芯片架构的拓扑结构。

Functions:
    create_chip_topology: 创建指定类型的芯片拓扑

Author: [待填写]
Created: 2025-10-21
"""

from typing import Literal
import numpy as np
import networkx as nx


TopologyType = Literal['path', 'cycle', 'grid', 'heavy_hex', 'kagome']


def create_chip_topology(size: int, topology_type: TopologyType) -> nx.Graph:
    """创建符合实际芯片的拓扑结构

    Parameters
    ----------
    size : int
        量子比特数量
    topology_type : str
        拓扑类型，支持:
        - 'path': 线形（一维链）
        - 'cycle': 环形（一维环）
        - 'grid': 二维网格
        - 'heavy_hex': 重六边形（IBM超导芯片拓扑）
        - 'kagome': Kagome晶格（三角-六边形结构）

    Returns
    -------
    nx.Graph
        生成的拓扑图

    Examples
    --------
    >>> G = create_chip_topology(6, 'grid')
    >>> print(G.number_of_nodes(), G.number_of_edges())
    6 7

    >>> G = create_chip_topology(5, 'heavy_hex')
    >>> print(G.number_of_nodes())
    5
    """
    if topology_type == 'path':
        # 线形拓扑：一维链
        return nx.path_graph(size)

    elif topology_type == 'cycle':
        # 环形拓扑：一维环
        return nx.cycle_graph(size)

    elif topology_type == 'grid':
        # 二维网格拓扑
        rows = int(np.sqrt(size))
        cols = (size + rows - 1) // rows
        G = nx.grid_2d_graph(rows, cols)
        nodes = list(G.nodes())[:size]
        G = G.subgraph(nodes).copy()
        # 转换为整数标签
        G = nx.convert_node_labels_to_integers(G)
        return G

    elif topology_type == 'heavy_hex':
        # IBM Heavy-Hex拓扑：基于六边形的结构
        # 这是IBM超导芯片的实际拓扑
        if size <= 5:
            # 小规模：简单的六边形单元
            G = nx.Graph()
            G.add_nodes_from(range(size))
            if size >= 2:
                G.add_edge(0, 1)
            if size >= 3:
                G.add_edges_from([(1, 2), (2, 0)])  # 形成三角形
            if size >= 4:
                G.add_edge(0, 3)
            if size >= 5:
                G.add_edges_from([(1, 4), (3, 4)])
        else:
            # 大规模：使用六边形晶格近似
            approx_side = int(np.ceil(np.sqrt(size / 1.5)))
            hex_graph = nx.hexagonal_lattice_graph(approx_side, approx_side)
            nodes = list(hex_graph.nodes())[:size]
            G = hex_graph.subgraph(nodes).copy()
            G = nx.convert_node_labels_to_integers(G)
        return G

    elif topology_type == 'kagome':
        # Kagome晶格：三角-六边形结构
        # 这种结构在某些拓扑量子计算设计中使用
        if size <= 6:
            # 小规模：手工构建Kagome单元
            G = nx.Graph()
            G.add_nodes_from(range(size))

            if size >= 3:
                # 中心三角形
                G.add_edges_from([(0, 1), (1, 2), (2, 0)])

            if size >= 6:
                # 外围三角形
                G.add_edges_from([
                    (0, 3), (1, 4), (2, 5),  # 连接到外围
                    (3, 4), (4, 5), (5, 3)   # 外围三角形
                ])
            elif size == 4:
                G.add_edge(0, 3)
            elif size == 5:
                G.add_edges_from([(0, 3), (1, 4)])
        else:
            # 大规模：使用三角晶格近似
            approx_side = int(np.ceil(np.sqrt(size / 1.2)))
            tri_graph = nx.triangular_lattice_graph(approx_side, approx_side)
            nodes = list(tri_graph.nodes())[:size]
            G = tri_graph.subgraph(nodes).copy()
            G = nx.convert_node_labels_to_integers(G)
        return G

    else:
        raise ValueError(
            f"Unknown topology type: {topology_type}. "
            f"Supported types: path, cycle, grid, heavy_hex, kagome"
        )


def get_topology_stats(G: nx.Graph) -> dict:
    """获取拓扑统计信息

    Parameters
    ----------
    G : nx.Graph
        拓扑图

    Returns
    -------
    dict
        统计信息字典
    """
    degrees = [G.degree(n) for n in G.nodes()]

    stats = {
        'num_nodes': G.number_of_nodes(),
        'num_edges': G.number_of_edges(),
        'avg_degree': np.mean(degrees),
        'max_degree': np.max(degrees) if degrees else 0,
        'min_degree': np.min(degrees) if degrees else 0,
        'is_connected': nx.is_connected(G),
        'density': nx.density(G)
    }

    if nx.is_connected(G):
        stats['diameter'] = nx.diameter(G)
        stats['avg_path_length'] = nx.average_shortest_path_length(G)
    else:
        stats['diameter'] = float('inf')
        stats['avg_path_length'] = float('inf')
        stats['num_components'] = nx.number_connected_components(G)

    return stats


if __name__ == "__main__":
    print("超导芯片拓扑生成器测试\n")
    print("=" * 70)

    test_size = 6
    topologies = ['path', 'cycle', 'grid', 'heavy_hex', 'kagome']

    print(f"测试规模: {test_size}个量子比特\n")

    for topo_type in topologies:
        G = create_chip_topology(test_size, topo_type)
        stats = get_topology_stats(G)

        print(f"{topo_type:12s}:")
        print(f"  节点数: {stats['num_nodes']}")
        print(f"  边数: {stats['num_edges']}")
        print(f"  平均度: {stats['avg_degree']:.2f}")
        print(f"  最大度: {stats['max_degree']}")
        print(f"  连通性: {'是' if stats['is_connected'] else '否'}")
        if stats['is_connected']:
            print(f"  平均路径: {stats['avg_path_length']:.2f}")
        print()

    print("=" * 70)
    print("\n拓扑说明:")
    print("  path:      线形拓扑（一维链）- 最简单结构")
    print("  cycle:     环形拓扑（一维环）- 周期边界")
    print("  grid:      二维网格拓扑 - 最常见的2D排列")
    print("  heavy_hex: 重六边形拓扑 - IBM超导芯片实际使用")
    print("  kagome:    Kagome晶格 - 高度连通的三角-六边形结构")
