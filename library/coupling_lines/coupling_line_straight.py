############################################################################################
# File Name: coupling_line_straight.py
# Description: This file primarily contains the code for constructing straight-type coupling lines.
############################################################################################

from addict import Dict
from base.library_base import LibraryBase
import toolbox
import copy, gdspy
import numpy as np
import os, sys
POJECT_ROOT =os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../")
if POJECT_ROOT not in sys.path:
    sys.path.append(POJECT_ROOT)
import gdsocc 
from OCC.Core import BRepAlgoAPI 

class CouplingLineStraight(LibraryBase):
    """
    CouplingLineStraight class for creating straight-type coupling line structures.

    Attributes:
        default_options: Dict, containing default coupling line parameters.
    """
    default_options = Dict(
        # Framework
        name="q0_cp_q1",
        type="CouplingLineStraight",
        chip='main',
        qubits=["q0", "q1"],
        outline=[],
        width=10,
        gap=5,
        start_pos=[0, 0],
        end_pos=[500, 0],
        # Geometric parameters
    )

    def __init__(self, options: Dict=None):
        """
        Initializes the CouplingLineStraight class.

        Input:
            options: Dict, user-defined coupling line parameters.

        Output:
            None.
        """
        super().__init__(options)
        return

    def calc_general_ops(self):
        """
        Calculates general operations.

        Output:
            None.
        """
        return

    def draw_gds(self):
        """
        Draws the geometric shape of CouplingLineStraight and adds it to the GDS cell.

        Output:
            None.
        """
        self.lib = gdspy.GdsLibrary()
        gdspy.library.use_current_library = False
        self.cell_subtract = self.lib.new_cell(self.name + "_subtract")

        path = gdspy.PolyPath([self.start_pos, self.end_pos], self.width)

        self.cell_subtract.add(path)

        self.cell_extract = self.lib.new_cell(self.name + "_extract")

        path = gdspy.PolyPath([self.start_pos, self.end_pos], self.width + self.gap*2)

        self.cell_extract.add(path)

        sub_poly = gdspy.boolean(self.cell_extract, self.cell_subtract, "not")
        self.cell = self.lib.new_cell(self.name + "_cell")
        self.cell.add(sub_poly)
        return
    
    def draw_shape(self):
        # FIXME:use gdspy.PolyPath like interface
        path_sub = gdsocc.Path(self.width, self.start_pos)
        path_sub.segment(500, "+x")
        # path_sub.done()

        path = gdsocc.Path(self.width + self.gap*2, self.start_pos)
        path.segment(500, "+x")
        # path.done()

        cut_op = BRepAlgoAPI.BRepAlgoAPI_Cut(path.shape, path_sub.shape)
        path.shape = cut_op.Shape()
        
        return path