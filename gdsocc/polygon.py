from OCC.Core import BRepBuilderAPI, BRepTools, TopExp, TopAbs, BRepAdaptor, TopoDS
from OCC.Core import gp, BRep, BRepAlgoAPI
from OCC.Core import BRepOffsetAPI
from OCC.Core import Geom, Geom2d, GeomAbs

import numpy


class ElementBase:
    def __init__(self, layer=0, datatype=0):
        self.layer = layer
        self.datatype = datatype
        self.shape = None


class Polygon(ElementBase):
    def __init__(self, points: list, layer=0, datatype=0):
        super().__init__(layer, datatype)
        if points:
            polygon = BRepBuilderAPI.BRepBuilderAPI_MakePolygon()
            for point in points:
                polygon.Add(gp.gp_Pnt(*point, 0))
            polygon.Close()
            self.shape = BRepBuilderAPI.BRepBuilderAPI_MakeFace(
                polygon.Wire()).Face()


class Rectangle(Polygon):
    def __init__(self, point1, point2, layer=0, datatype=0):
        points = [[point1[0], point1[1]],
                  [point1[0], point2[1]],
                  [point2[0], point2[1]],
                  [point2[0], point1[1]]]
        super().__init__(points, layer, datatype)


class Path(Polygon):
    def __init__(self, width, initial_point=(0, 0), number_of_paths=1, distance=0):
        super().__init__([])
        self.x = initial_point[0]
        self.y = initial_point[1]
        self.half_width = width * 0.5
        self.shape = TopoDS.TopoDS_Compound()
        self.builder = BRep.BRep_Builder()
        self.builder.MakeCompound(self.shape)

    def segment(self, length, direction):
        if direction == "+x":
            ca = 1
            sa = 0
        elif direction == "-x":
            ca = -1
            sa = 0
        elif direction == "+y":
            ca = 0
            sa = 1
        elif direction == "-y":
            ca = 0
            sa = -1
        else:
            ca = numpy.cos(direction)
            sa = numpy.sin(direction)

        next_x = self.x + length * ca
        next_y = self.y + length * sa

        edge = BRepBuilderAPI.BRepBuilderAPI_MakeEdge(
            gp.gp_Pnt(self.x, self.y, 0), gp.gp_Pnt(next_x, next_y, 0)).Edge()

        self.x = next_x
        self.y = next_y

        wire = BRepBuilderAPI.BRepBuilderAPI_MakeWire(edge).Wire()

        # get first point coord and tangent of new edge
        adaptor = BRepAdaptor.BRepAdaptor_CompCurve(wire)
        first_param = adaptor.FirstParameter()
        tangent_start = gp.gp_Vec()
        p_start = gp.gp_Pnt()
        adaptor.D1(first_param, p_start, tangent_start)

        # make local axis in wire start end point and tangent vec
        axis_global = gp.gp_Ax3(
            gp.gp_Pnt(0, 0, 0), gp.gp_Dir(0, 0, 1), gp.gp_Dir(1, 0, 0))
        dir_start = gp.gp_Dir(tangent_start)
        axis_start = gp.gp_Ax3(p_start, gp.gp_Dir(0, 0, 1), dir_start)

        # make a sweep profile line on local axis
        local_start_1 = gp.gp_Pnt(0, +self.half_width, 0)
        local_start_2 = gp.gp_Pnt(0, -self.half_width, 0)
        # translate local point to global axis
        trsf_start = gp.gp_Trsf()
        trsf_start.SetTransformation(axis_start, axis_global)

        norm_line = BRepBuilderAPI.BRepBuilderAPI_MakeEdge(local_start_1.Transformed(
            trsf_start), local_start_2.Transformed(trsf_start)).Edge()

        pipe = BRepOffsetAPI.BRepOffsetAPI_MakePipe(wire, norm_line).Shape()
        self.builder.Add(self.shape, pipe)

    # def done(self):
    #     # get tangent in wire start and end point
    #     adaptor = BRepAdaptor.BRepAdaptor_CompCurve(self.spine.Wire())
    #     first_param = adaptor.FirstParameter()
    #     last_param = adaptor.LastParameter()

    #     tangent_start = gp.gp_Vec()
    #     tangent_end = gp.gp_Vec()
    #     p_start = gp.gp_Pnt()
    #     p_end = gp.gp_Pnt()

    #     adaptor.D1(first_param, p_start, tangent_start)
    #     adaptor.D1(last_param, p_end, tangent_end)

    #     dir_start = gp.gp_Dir(tangent_start)
    #     dir_end = gp.gp_Dir(tangent_end)

    #     # make local axis in wire start end point and tangent vec
    #     axis_global = gp.gp_Ax3(
    #         gp.gp_Pnt(0, 0, 0), gp.gp_Dir(0, 0, 1), gp.gp_Dir(1, 0, 0))
    #     axis_start = gp.gp_Ax3(p_start, gp.gp_Dir(0, 0, 1), dir_start)
    #     axis_end = gp.gp_Ax3(p_end, gp.gp_Dir(0, 0, 1), dir_end)

    #     # make a local box to cut off extend wire start and end
    #     # box points in local axis
    #     local_start_1 = gp.gp_Pnt(0, -self.half_width, 0)
    #     local_start_2 = gp.gp_Pnt(0, +self.half_width, 0)
    #     local_start_3 = gp.gp_Pnt(-self.half_width, +self.half_width, 0)
    #     local_start_4 = gp.gp_Pnt(-self.half_width, -self.half_width, 0)
    #     # translate local point to global axis
    #     trsf_start = gp.gp_Trsf()
    #     trsf_start.SetTransformation(axis_start, axis_global)
    #     polygon_start = BRepBuilderAPI.BRepBuilderAPI_MakePolygon()
    #     polygon_start.Add(local_start_1.Transformed(trsf_start))
    #     polygon_start.Add(local_start_2.Transformed(trsf_start))
    #     polygon_start.Add(local_start_3.Transformed(trsf_start))
    #     polygon_start.Add(local_start_4.Transformed(trsf_start))
    #     polygon_start.Close()
    #     face_start = BRepBuilderAPI.BRepBuilderAPI_MakeFace(
    #         polygon_start.Wire()).Face()

    #     # do same thing for end point
    #     local_end_1 = gp.gp_Pnt(0, -self.half_width, 0)
    #     local_end_2 = gp.gp_Pnt(0, +self.half_width, 0)
    #     local_end_3 = gp.gp_Pnt(self.half_width, +self.half_width, 0)
    #     local_end_4 = gp.gp_Pnt(self.half_width, -self.half_width, 0)
    #     # translate local point to global axis
    #     trsf_end = gp.gp_Trsf()
    #     trsf_end.SetTransformation(axis_end, axis_global)
    #     polygon_end = BRepBuilderAPI.BRepBuilderAPI_MakePolygon()
    #     polygon_end.Add(local_end_1.Transformed(trsf_end))
    #     polygon_end.Add(local_end_2.Transformed(trsf_end))
    #     polygon_end.Add(local_end_3.Transformed(trsf_end))
    #     polygon_end.Add(local_end_4.Transformed(trsf_end))
    #     polygon_end.Close()
    #     face_end = BRepBuilderAPI.BRepBuilderAPI_MakeFace(
    #         polygon_end.Wire()).Face()

    #     # BRepOffsetAPI_MakeOffset require wire must be closed
    #     # so we add reverse edges of wire to close it
    #     '''
    #     exp = TopExp.TopExp_Explorer(self.spine.Wire(), TopAbs.TopAbs_EDGE)
    #     reversed_edge = []
    #     while exp.More():
    #         edge = exp.Current()
    #         r_edge = edge.Reversed()
    #         reversed_edge.append(r_edge)
    #         exp.Next()
    #     for i in reversed(reversed_edge):
    #         self.spine.Add(i)
    #     '''
    #     wire = self.spine.Wire()
    #     offset = BRepOffsetAPI.BRepOffsetAPI_MakeOffset(
    #         wire, GeomAbs.GeomAbs_JoinType.GeomAbs_Intersection)
    #     offset.Perform(self.half_width)
    #     polygon = BRepBuilderAPI.BRepBuilderAPI_MakeFace(
    #         offset.Shape()).Face()
    #     # polygon = offset.Shape()
    #     # as GeomAbs.GeomAbs_Arc make start/end point a arc, we cut off it
    #     opt_start = BRepAlgoAPI.BRepAlgoAPI_Cut(polygon, face_start)
    #     polygon = opt_start.Shape()
    #     opt_start = BRepAlgoAPI.BRepAlgoAPI_Cut(polygon, face_end)
    #     polygon = opt_start.Shape()
    #     self.shape = polygon


if __name__ == "__main__":
    cpw = Path(40, [0.5, 0.5])
    cpw.segment(1000, direction="+x")
    cpw.done()
