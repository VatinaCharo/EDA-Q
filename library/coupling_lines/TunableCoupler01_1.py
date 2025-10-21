
import gdspy, copy
from addict import Dict
import math as mt
from base.library_base import LibraryBase

class Tunablecoupler011(LibraryBase):
    default_options = Dict(
        name="Tunablecoupler0110",
        type="Tunablecoupler011",
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

            ([[-0.19999999999999998, -0.01], [-0.19999999999999998, 0.01], [-0.19999999999999998, 0.06], [-0.18, 0.06], [-0.18, 0.01], [-0.124, 0.01], [-0.124, 0.06], [-0.104, 0.06], [-0.104, 0.01], [-0.048, 0.01], [-0.048, 0.06], [-0.027999999999999997, 0.06], [-0.027999999999999997, 0.01], [0.0, 0.01], [0.027999999999999997, 0.01], [0.027999999999999997, 0.06], [0.048, 0.06], [0.048, 0.01], [0.104, 0.01], [0.104, 0.06], [0.124, 0.06], [0.124, 0.01], [0.18, 0.01], [0.18, 0.06], [0.19999999999999998, 0.06], [0.19999999999999998, 0.01], [0.19999999999999998, -0.01], [0.0, -0.01], [-0.19999999999999998, -0.01]], 1),

            ([[0.003, 0.048999999999999995], [-0.003, 0.048999999999999995], [-0.003, 0.079], [-0.003, 0.094], [0.079, 0.094], [0.079, 0.079], [0.079, 0.048999999999999995], [0.073, 0.048999999999999995], [0.073, 0.079], [0.003, 0.079], [0.003, 0.048999999999999995]], 1),

            ([[-0.124, 0.08], [-0.124, 0.075], [-0.1115, 0.075], [-0.1115, 0.075], [-0.1115, 0.0875], [-0.11649999999999999, 0.0875], [-0.11649999999999999, 0.08]], 1),

            ([[0.252, 0.11199999999999999], [-0.252, 0.11199999999999999], [-0.252, 0.072], [-0.124, 0.072], [-0.124, 0.08299999999999999], [-0.1195, 0.08299999999999999], [-0.1195, 0.0875], [-0.1085, 0.0875], [-0.1085, 0.072], [-0.124, 0.072], [-0.252, 0.072], [-0.252, -0.02], [-0.21, -0.02], [-0.21, 0.06999999999999999], [-0.16999999999999998, 0.06999999999999999], [-0.16999999999999998, 0.02], [-0.13399999999999998, 0.02], [-0.13399999999999998, 0.06999999999999999], [-0.094, 0.06999999999999999], [-0.094, 0.02], [-0.057999999999999996, 0.02], [-0.057999999999999996, 0.06999999999999999], [-0.018, 0.06999999999999999], [-0.018, 0.043], [-0.009, 0.043], [-0.009, 0.09999999999999999], [0.08499999999999999, 0.09999999999999999], [0.08499999999999999, 0.043], [0.06699999999999999, 0.043], [0.06699999999999999, 0.073], [0.009, 0.073], [0.009, 0.043], [-0.009, 0.043], [-0.018, 0.043], [-0.018, 0.02], [0.018, 0.02], [0.018, 0.06999999999999999], [0.057999999999999996, 0.06999999999999999], [0.057999999999999996, 0.02], [0.094, 0.02], [0.094, 0.06999999999999999], [0.13399999999999998, 0.06999999999999999], [0.13399999999999998, 0.02], [0.16999999999999998, 0.02], [0.16999999999999998, 0.06999999999999999], [0.21, 0.06999999999999999], [0.21, -0.02], [-0.21, -0.02], [-0.252, -0.02], [-0.252, -0.032], [0.252, -0.032]], 1),

            ([[-0.235, 0.095], [-0.159, 0.095], [-0.149, 0.095], [-0.149, 0.108], [-0.1445, 0.108], [-0.1445, 0.11249999999999999], [-0.08349999999999999, 0.11249999999999999], [-0.08349999999999999, 0.095], [-0.08299999999999999, 0.095], [-0.06899999999999999, 0.095], [-0.033999999999999996, 0.095], [-0.033999999999999996, 0.125], [0.11, 0.125], [0.11, 0.095], [0.145, 0.095], [0.159, 0.095], [0.235, 0.095], [0.235, -0.045], [-0.235, -0.045], [-0.235, 0.095]], 1),

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
