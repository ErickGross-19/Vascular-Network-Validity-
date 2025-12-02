"""
Vascular Design Library - LLM-Friendly API for Iterative Vascular Network Design

A composable "LEGO kit" library that enables LLMs to design vascular networks
through small, explicit operations with structured feedback.

Key Features:
- Small, composable operations (verbs)
- Structured results with status, IDs, and warnings
- Full serializability (JSON/dict)
- Deterministic randomness
- Reversible changes (undo/redo support)

Example Usage:
    from vascular_lib import create_network, add_inlet, grow_branch
    from vascular_lib.core import EllipsoidDomain
    
    domain = EllipsoidDomain(0.12, 0.10, 0.08)
    network = create_network(domain, seed=42)
    
    result = add_inlet(network, position=(0, 0, 0), direction=(1, 0, 0), radius=0.005)
    inlet_id = result.new_ids['node']
    
    result = grow_branch(network, from_node_id=inlet_id, length=0.01, direction=(1, 0, 0))
"""

__version__ = "1.0.0"

from .core.types import Point3D, Direction3D, TubeGeometry
from .core.network import Node, VesselSegment, VascularNetwork
from .core.domain import DomainSpec, EllipsoidDomain, MeshDomain
from .core.result import OperationResult, Delta

from .ops.build import create_network, add_inlet, add_outlet

from .ops.growth import grow_branch, bifurcate
from .ops.space_colonization import space_colonization_step

from .analysis.query import (
    get_leaf_nodes,
    get_paths_from_inlet,
    get_branch_order,
    measure_segment_lengths,
)
from .analysis.coverage import compute_coverage
from .analysis.flow import estimate_flows, check_hemodynamic_plausibility

from .ops.collision import get_collisions, avoid_collisions

from .io.serialize import save_json, load_json

__all__ = [
    "Point3D",
    "Direction3D",
    "TubeGeometry",
    "Node",
    "VesselSegment",
    "VascularNetwork",
    "DomainSpec",
    "EllipsoidDomain",
    "MeshDomain",
    "OperationResult",
    "Delta",
    "create_network",
    "add_inlet",
    "add_outlet",
    "grow_branch",
    "bifurcate",
    "space_colonization_step",
    "get_leaf_nodes",
    "get_paths_from_inlet",
    "get_branch_order",
    "measure_segment_lengths",
    "compute_coverage",
    "estimate_flows",
    "check_hemodynamic_plausibility",
    "get_collisions",
    "avoid_collisions",
    "save_json",
    "load_json",
]
