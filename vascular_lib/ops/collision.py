"""
Collision detection and avoidance operations.
"""

from typing import List, Tuple
from ..core.network import VascularNetwork
from ..core.result import OperationResult, OperationStatus


def get_collisions(
    network: VascularNetwork,
    min_clearance: float = 0.001,
    exclude_connected: bool = True,
) -> OperationResult:
    """
    Find all segment pairs that are too close.
    
    Parameters
    ----------
    network : VascularNetwork
        Network to check
    min_clearance : float
        Minimum required clearance between segments (meters)
    exclude_connected : bool
        If True, exclude segments that share a node
    
    Returns
    -------
    result : OperationResult
        Result with metadata['collisions'] containing list of collision tuples
    """
    spatial_index = network.get_spatial_index()
    
    collisions = spatial_index.get_collisions(
        min_clearance=min_clearance,
        exclude_connected=exclude_connected,
    )
    
    if collisions:
        return OperationResult(
            status=OperationStatus.WARNING,
            message=f"Found {len(collisions)} collisions",
            warnings=[f"Segments {c[0]} and {c[1]} too close (distance: {c[2]:.4f}m)" for c in collisions[:5]],
            metadata={"collisions": collisions, "count": len(collisions)},
        )
    else:
        return OperationResult.success(
            message="No collisions found",
            metadata={"collisions": [], "count": 0},
        )


def avoid_collisions(
    network: VascularNetwork,
    min_clearance: float = 0.001,
    repair_strategy: str = "report",
) -> OperationResult:
    """
    Detect and optionally repair collisions.
    
    Parameters
    ----------
    network : VascularNetwork
        Network to check/repair
    min_clearance : float
        Minimum required clearance
    repair_strategy : str
        Strategy for handling collisions:
        - "report": Just report collisions
        - "terminate": Mark colliding branches as terminated (future work)
        - "reroute": Attempt to reroute branches (future work)
    
    Returns
    -------
    result : OperationResult
        Result with collision information
    """
    if repair_strategy != "report":
        return OperationResult.failure(
            message=f"Repair strategy '{repair_strategy}' not yet implemented",
            errors=["Only 'report' strategy is currently supported"],
        )
    
    return get_collisions(network, min_clearance=min_clearance)
