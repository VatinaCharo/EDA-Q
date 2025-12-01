# EDA-Q 演示示例代码
# 用于展示AI助手的上下文理解和代码优化功能

import sys
sys.path.append("../../")
from api.design import Design
from addict import Dict

# 示例1: 基础设计流程
def basic_design_example():
    """16比特芯片基础设计"""
    design = Design()
    design.generate_topology(qubits_num=16)
    design.topology.generate_full_edges()
    design.generate_qubits(topology=True, qubits_type="Transmon", dist=2000)
    design.generate_chip(qubits=True, chip_name="chip0")
    design.copy_chip(old_chip_name="chip0", new_chip_name="chip1")
    design.generate_coupling_lines(topology=True, qubits=True, cpls_type="CouplerBase")
    design.routing(method="Flipchip_routing", chip_name="chip1")
    design.gds.save_gds("basic_chip.gds")

# 示例2: 可优化的代码(演示AI代码优化功能)
def example_to_optimize():
    """这段代码可以优化"""
    design = Design()
    design.generate_topology(qubits_num=16)
    design.topology.generate_full_edges()
    design.topology.show_image()
    design.generate_qubits(topology=True, qubits_type="Transmon", dist=2000)
    design.gds.show_svg()
    design.generate_chip(qubits=True, chip_name="chip0")
    design.gds.show_svg()
    # 可以优化: 减少重复的show_svg调用

# 示例3: 自定义拓扑
def custom_topology_example():
    """自定义拓扑结构"""
    design = Design(topo_row=5, topo_col=5)
    # 手动添加边
    design.topology.add_edge("q0", "q1")
    design.topology.add_edge("q1", "q2")
    design.topology.add_edge("q5", "q6")
    # 批量添加
    design.topology.batch_add_edges_list(y=[0, 2, 4])
    design.topology.show_image()

# 示例4: 完整流程(带读取腔)
def complete_chip_example():
    """完整的芯片设计流程"""
    design = Design()

    # 1. 拓扑
    design.generate_topology(qubits_num=9)
    design.topology.generate_full_edges()

    # 2. 量子比特
    design.generate_qubits(
        topology=True,
        qubits_type="Transmon",
        dist=2500,
        chip_name="chip0"
    )

    # 3. 芯片层
    design.generate_chip(qubits=True, dist=4000, chip_name="chip0")
    design.copy_chip(old_chip_name="chip0", new_chip_name="chip1")

    # 4. 耦合器
    design.generate_coupling_lines(
        topology=True,
        qubits=True,
        cpls_type="CouplingCavity"
    )

    # 5. 读取腔
    design.generate_readout_lines(
        qubits=True,
        rdls_type="ReadoutCavity",
        chip_name="chip0"
    )

    # 6. 布线
    design.routing(
        method="Flipchip_routing_IBM",
        chip_name="chip1"
    )

    # 7. 保存
    design.gds.save_gds("complete_chip.gds")
    design.gds.show_svg()

if __name__ == "__main__":
    # 运行示例
    print("选择要运行的示例:")
    print("1. 基础设计")
    print("2. 可优化代码")
    print("3. 自定义拓扑")
    print("4. 完整流程")

    choice = input("输入选项(1-4): ")

    if choice == "1":
        basic_design_example()
    elif choice == "2":
        example_to_optimize()
    elif choice == "3":
        custom_topology_example()
    elif choice == "4":
        complete_chip_example()
    else:
        print("无效选项")
