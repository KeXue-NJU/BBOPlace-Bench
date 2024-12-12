
REGISTRY = {}

# from .grid_placer import GridPlacer
from .grid_guide_placer import GridGuidePlacer
from .sp_placer import SPPlacer
from .bstar_tree_placer import BstarTreePlacer
from .dmp_placer import DMPPlacer

# REGISTRY["grid"] = GridPlacer
REGISTRY["grid_guide"] = GridGuidePlacer
REGISTRY["sp"] = SPPlacer
REGISTRY["bstar_tree"] = BstarTreePlacer
REGISTRY["dmp"] = DMPPlacer