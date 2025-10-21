
import gdspy, copy
from addict import Dict
import math as mt
from base.library_base import LibraryBase

class Circleraster1(LibraryBase):
    default_options = Dict(
        name="Circleraster10",
        type="Circleraster1",
        chip="chip0",
        gds_pos=(0, 0),
        topo_pos=(0, 0),
        outline=[],
        rotation=0
    )

    def __init__(self, options: Dict = None):
        super().__init__(options)
        return

    def _add_polygons_to_cell(self):
        polygon_data = [

            ([[0.3, 0.0], [0.29855499999999996, -0.029404999999999997], [0.294236, -0.058526999999999996], [0.287082, -0.087085], [0.27716399999999997, -0.11480499999999999], [0.264576, -0.141419], [0.249441, -0.16667099999999999], [0.231903, -0.190318], [0.212132, -0.212132], [0.190318, -0.231903], [0.16667099999999999, -0.249441], [0.141419, -0.264576], [0.11480499999999999, -0.27716399999999997], [0.087085, -0.287082], [0.058526999999999996, -0.294236], [0.029404999999999997, -0.29855499999999996], [0.0, -0.3], [-0.029404999999999997, -0.29855499999999996], [-0.058526999999999996, -0.294236], [-0.087085, -0.287082], [-0.11480499999999999, -0.27716399999999997], [-0.141419, -0.264576], [-0.16667099999999999, -0.249441], [-0.190318, -0.231903], [-0.212132, -0.212132], [-0.231903, -0.190318], [-0.249441, -0.16667099999999999], [-0.264576, -0.141419], [-0.27716399999999997, -0.11480499999999999], [-0.287082, -0.087085], [-0.294236, -0.058526999999999996], [-0.29855499999999996, -0.029404999999999997], [-0.3, 0.0], [-0.29855499999999996, 0.029404999999999997], [-0.294236, 0.058526999999999996], [-0.287082, 0.087085], [-0.27716399999999997, 0.11480499999999999], [-0.264576, 0.141419], [-0.249441, 0.16667099999999999], [-0.231903, 0.190318], [-0.212132, 0.212132], [-0.190318, 0.231903], [-0.16667099999999999, 0.249441], [-0.141419, 0.264576], [-0.11480499999999999, 0.27716399999999997], [-0.087085, 0.287082], [-0.058526999999999996, 0.294236], [-0.029404999999999997, 0.29855499999999996], [0.0, 0.3], [0.029404999999999997, 0.29855499999999996], [0.058526999999999996, 0.294236], [0.087085, 0.287082], [0.11480499999999999, 0.27716399999999997], [0.141419, 0.264576], [0.16667099999999999, 0.249441], [0.190318, 0.231903], [0.212132, 0.212132], [0.231903, 0.190318], [0.249441, 0.16667099999999999], [0.264576, 0.141419], [0.27716399999999997, 0.11480499999999999], [0.287082, 0.087085], [0.294236, 0.058526999999999996], [0.29855499999999996, 0.029404999999999997], [0.3, 0.0]], 1),

        ]

        for points in polygon_data:
            layer = points[-1]
            polygon_points = points[0]
            self.cell.add(gdspy.Polygon(polygon_points, layer=layer))

    def _calculate_bounding_box(self):
        min_x, min_y = float("inf"), float("inf")
        max_x, max_y = float("-inf"), float("-inf")

        for polygon in self.cell.polygons:
            bbox = polygon.get_bounding_box()
            if bbox is not None:
                min_x = min(min_x, bbox[0][0])
                min_y = min(min_y, bbox[0][1])
                max_x = max(max_x, bbox[1][0])
                max_y = max(max_y, bbox[1][1])

        if min_x == float("inf") or min_y == float("inf"):
            raise ValueError("No valid polygons found in the cell.")

        return min_x, min_y, max_x, max_y

    def _transform_polygons(self, dx, dy, rotation, center):
        for polygon in self.cell.polygons:
            polygon.translate(dx, dy)
            polygon.rotate(rotation, center=center)

    def calc_general_ops(self):
        self.lib = gdspy.GdsLibrary()
        gdspy.library.use_current_library = False
        self.cell = self.lib.new_cell(self.name + "_cell")

        self._add_polygons_to_cell()

        # Bounding box calculations
        min_x, min_y, max_x, max_y = self._calculate_bounding_box()
        current_center_x = (min_x + max_x) / 2
        current_center_y = (min_y + max_y) / 2

        # Target center
        target_center_x, target_center_y = self.options.gds_pos
        dx = target_center_x - current_center_x
        dy = target_center_y - current_center_y

        # Update gds_pos to match adjusted center
        self.options.gds_pos = (target_center_x, target_center_y)

        # Apply transformations
        self._transform_polygons(dx, dy, self.rotation, center=(target_center_x, target_center_y))
        self.outline = [
            points for polygon in self.cell.polygons for points in polygon.get_bounding_box().tolist()
        ]
        return

    def draw_gds(self):
        self.calc_general_ops()
