"""NuScenes-to-NuRec dynamic-track compatibility contract.

Keep this module dependency-free so the mapping can be checked without the
NuScenes or NCore runtimes.  The values intentionally match the label classes
in NVIDIA NRE 26.04's ``car2sim_6cam`` recipe.
"""

from __future__ import annotations

from typing import Dict, FrozenSet


# nuScenes annotations are human-created dataset annotations.  They remain
# EXTERNAL instead of being relabelled as AUTOLABEL merely to satisfy a recipe.
NUREC_TRACK_LABEL_SOURCE = "EXTERNAL"

NUSCENES_CATEGORY_MAP: Dict[str, str] = {
    "vehicle.car": "automobile",
    "vehicle.truck": "heavy_truck",
    "vehicle.bus.bendy": "bus",
    "vehicle.bus.rigid": "bus",
    "vehicle.construction": "Other Vehicle - Construction Vehicle",
    "vehicle.motorcycle": "motorcycle",
    "vehicle.bicycle": "bicycle",
    "vehicle.trailer": "trailer",
    "vehicle.emergency.ambulance": "Emergency Vehicle",
    "vehicle.emergency.police": "Emergency Vehicle",
    "human.pedestrian.adult": "pedestrian",
    "human.pedestrian.child": "pedestrian",
    "human.pedestrian.construction_worker": "pedestrian",
    "human.pedestrian.police_officer": "pedestrian",
    # These are retained in NCore for provenance and future static-object use.
    # The NRE 26.04 dynamic layers do not consume either class by default.
    "movable_object.barrier": "barrier",
    "movable_object.trafficcone": "traffic_cone",
}

NUREC_DYNAMIC_RIGID_CLASSES: FrozenSet[str] = frozenset(
    {
        "automobile",
        "heavy_truck",
        "bus",
        "Other Vehicle - Construction Vehicle",
        "trailer",
        "Emergency Vehicle",
    }
)
NUREC_DYNAMIC_DEFORMABLE_CLASSES: FrozenSet[str] = frozenset(
    {"pedestrian", "motorcycle", "bicycle"}
)

