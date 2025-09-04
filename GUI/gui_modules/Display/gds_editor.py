from PyQt5.QtWidgets import QApplication, QWidget, QMenu, QFileDialog
from PyQt5.QtCore import Qt, QSize, pyqtSignal, QPoint, QStandardPaths

from OCC.Display.OCCViewer import Viewer3d
from OCC.Core import gp, BRepBuilderAPI, Quantity, AIS
from OCC.Core import STEPControl

import networkx as nx

from enum import IntEnum


class GdsEditor(QWidget):

    class Action(IntEnum):
        SHOW_IN_INTERPRETER = 0
        SAVE_FILE = 1

    sendObjToInterpreter = pyqtSignal(dict)

    class RightButtonMenu(QMenu):
        """
        Menu for right click
        """
        action_triggered = pyqtSignal(int, QPoint)

        def __init__(self, parent=None):
            QMenu.__init__(self, parent)
            self.cur_pos = None
            self.addAction("Show in interpreter").triggered.connect(
                lambda check, self=self: self.action_triggered.emit(GdsEditor.Action.SHOW_IN_INTERPRETER.value, self.cur_pos))
            self.addAction("Save current layout to file").triggered.connect(
                lambda check, self=self: self.action_triggered.emit(GdsEditor.Action.SAVE_FILE.value, self.cur_pos))

        def popup(self, p, action=None):
            self.cur_pos = p
            return super().popup(p, action)

    def __init__(self, parent=None):
        QWidget.__init__(self, parent)
        # enable Mouse Tracking
        self.setMouseTracking(True)

        # Strong focus
        self.setFocusPolicy(Qt.FocusPolicy.WheelFocus)

        self.setAttribute(Qt.WidgetAttribute.WA_NativeWindow)
        self.setAttribute(Qt.WidgetAttribute.WA_PaintOnScreen)
        self.setAttribute(Qt.WidgetAttribute.WA_NoSystemBackground)

        self.setAutoFillBackground(False)

        self.right_menu = GdsEditor.RightButtonMenu(self)
        self.right_menu.action_triggered.connect(self.handleMenuAction)

        # use digraph to store component and shape relation
        self.component_shape_map = nx.DiGraph()

        # occ 3dviewer
        self._display = Viewer3d()
        self._display.Create(window_handle=int(
            self.winId()), parent=self, create_default_lights=False, display_glinfo=False)
        # background gradient
        self._display.SetModeShaded()
        # fix top view
        self._display.View_Top()
        self._display.set_bg_gradient_color([0, 10, 10], [0, 10, 10])

        self._pan_start_x = None
        self._pan_start_y = None
        self._drawbox = False

        self._move_start_x = None
        self._move_start_y = None

        self._selected_shape = None

    def sizeHint(self):
        return QSize(800, 600)

    def handleMenuAction(self, action, pos):
        '''
        process right mouse button menu action
        '''
        lpos = self.mapFromGlobal(pos)

        if action == GdsEditor.Action.SHOW_IN_INTERPRETER:
            self._display.Select(lpos.x(), lpos.y())
            selected_shape = self._display.Context.SelectedInteractive()
            if selected_shape is not None:
                self._display.Context.ClearSelected(True)
                ais_shape = AIS.AIS_Shape.DownCast(selected_shape)
                component = list(
                    self.component_shape_map.predecessors(ais_shape))[0]
                self.sendObjToInterpreter.emit({"selec_component": component})
        elif action == GdsEditor.Action.SAVE_FILE:
            save_fn, _ = QFileDialog.getSaveFileName(
                None, "Save File", QStandardPaths.writableLocation(QStandardPaths.StandardLocation.HomeLocation), ".step")
            if save_fn:
                self.saveCurrentLayout(save_fn)
            else:
                return

    def saveCurrentLayout(self, fn):
        writer = STEPControl.STEPControl_Writer()
        for node in self.component_shape_map:
            if type(node) == AIS.AIS_Shape:
                writer.Transfer(
                    node.Shape(), STEPControl.STEPControl_StepModelType.STEPControl_AsIs)

        writer.Write(fn)

    def resizeEvent(self, event):
        super(GdsEditor, self).resizeEvent(event)
        self._display.View.MustBeResized()

    def paintEngine(self):
        return None

    def keyPressEvent(self, event):
        super(GdsEditor, self).keyPressEvent(event)

    def focusInEvent(self, event):
        self._display.Repaint()

    def focusOutEvent(self, event):
        self._display.Repaint()

    def paintEvent(self, event):
        self._display.Context.UpdateCurrentViewer()

    def wheelEvent(self, event):
        delta = event.angleDelta().y()
        zoom_factor = 1.25 if delta > 0 else 0.75
        self._display.ZoomFactor(zoom_factor)

    def mousePressEvent(self, event):
        pos = event.pos()
        button = event.button()
        # use left mouse button to pan
        if button == Qt.MouseButton.LeftButton:
            self._pan_start_x = pos.x()
            self._pan_start_y = pos.y()
        elif button == Qt.MouseButton.MiddleButton:
            self._move_start_x = pos.x()
            self._move_start_y = pos.y()
            self._display.Context.ClearSelected(True)
            self._display.Select(pos.x(), pos.y())
            self._selected_shape = self._display.Context.SelectedInteractive()
        elif button == Qt.MouseButton.RightButton:
            gpos = self.mapToGlobal(pos)
            self.right_menu.popup(gpos)

    def mouseMoveEvent(self, event):
        pos = event.pos()
        buttons = event.buttons()
        # use left mouse button to pan
        if buttons == Qt.MouseButton.LeftButton:
            dx = pos.x() - self._pan_start_x
            dy = pos.y() - self._pan_start_y
            self._pan_start_x = pos.x()
            self._pan_start_y = pos.y()
            self._display.Pan(dx, -dy)
        if buttons == Qt.MouseButton.MiddleButton and self._selected_shape:
            # translate view coord to real coord
            view = self._display.View
            v1 = view.Convert(self._move_start_x, self._move_start_y)
            v2 = view.Convert(pos.x(), pos.y())
            move_vec = gp.gp_Vec(
                gp.gp_Pnt(v1[0], v1[1], 0), gp.gp_Pnt(v2[0], v2[1], 0))
            trsf = gp.gp_Trsf()
            trsf.SetTranslation(move_vec)
            self._selected_shape.SetLocalTransformation(
                self._selected_shape.Transformation()*trsf)
            self._display.Context.Update(self._selected_shape, True)

            self._move_start_x = pos.x()
            self._move_start_y = pos.y()
        else:
            self._display.MoveTo(pos.x(), pos.y())

    def mouseReleaseEvent(self, event):
        button = event.button()
        if button == Qt.MouseButton.MiddleButton:
            self._selected_shape = None
            self._display.Context.ClearSelected(True)

    def showTopoDS(self, component: list):
        for i in component:
            for s in i.shapes:
                self._display.DisplayShape(
                    s, color=Quantity.Quantity_NameOfColor.Quantity_NOC_CORAL, transparency=0.8)
        self._display.FitAll()

    def showComponents(self, components: list):
        for i in components:
            shape = i.draw_shape().shape
            ais_shapes = self._display.DisplayShape(
                shape, color=Quantity.Quantity_NameOfColor.Quantity_NOC_CORAL, transparency=0.8)
            for a in ais_shapes:
                self.component_shape_map.add_edge(i, a)
        self._display.FitAll()

    def updateComponent(self, comp):
        if comp in self.component_shape_map:
            for shape in self.component_shape_map.successors(comp):
                # save trans
                trans = shape.Transformation()
                self._display.Context.Erase(shape, False)

            for shape in comp.draw_shape().shapes:
                ais_shapes = self._display.DisplayShape(
                    shape, color=Quantity.Quantity_NameOfColor.Quantity_NOC_CORAL, transparency=0.8)
                for a in ais_shapes:
                    self.component_shape_map.add_edge(comp, a)
                    # apply transformation to new shape
                    a.SetLocalTransformation(trans)
            self._display.FitAll()

    def drawSelectBox(self, event):
        tolerance = 2
        pt = event.pos()
        dx = pt.x() - self.dragStartPosX
        dy = pt.y() - self.dragStartPosY
        if abs(dx) <= tolerance and abs(dy) <= tolerance:
            return
        self._drawbox = [self.dragStartPosX, self.dragStartPosY, dx, dy]

    def saveImage(self, path: str = None):
        self._display.ExportToImage(path)


if __name__ == "__main__":
    import sys
    import os
    from PyQt5.QtWidgets import QVBoxLayout, QMainWindow
    from OCC.Core.BRepPrimAPI import BRepPrimAPI_MakeBox
    from OCC.Core import BRepOffsetAPI, GeomAbs
    current_path = os.path.dirname(os.path.abspath(__file__))
    PROJ_PATH = os.path.abspath(os.path.join(
        os.path.dirname(current_path), "../.."))
    sys.path.append(PROJ_PATH)
    from library.qubits import transmon
    from library.coupling_lines import coupling_line_straight
    from GUI.gui_modules.Command.Python_Interpreter import PythonInterpreter
    import gdsocc
    # from api.design import Design

    # design = Design()
    # design.generate_topology(topo_col=32, topo_row=32)
    # design.topology.generate_random_edges(edges_num=1500)
    # design.generate_qubits(
    #     topology=True, qubits_type="Transmon", chip_name="chip0", dist=3000)
    # design.gds.show_svg()

    app = QApplication([])
    main_w = QMainWindow()

    vlayout = QVBoxLayout()
    main_widget = QWidget()
    main_widget.setLayout(vlayout)

    main_w.setCentralWidget(main_widget)

    viewer = GdsEditor()
    viewer.setBaseSize
    vlayout.addWidget(viewer)
    qubit = transmon.Transmon({})
    cpl = coupling_line_straight.CouplingLineStraight({})

    # viewer.showTopoDS([qubit.draw_shape(), cpl.draw_shape()])
    viewer.showComponents([qubit, cpl])

    pyinp = PythonInterpreter()
    pyinp.addLocalsVar({"viewer":  viewer})
    pyinp.clear_output()
    viewer.sendObjToInterpreter.connect(pyinp.addLocalsVar)
    vlayout.addWidget(pyinp)
    main_w.show()
    sys.exit(app.exec_())
