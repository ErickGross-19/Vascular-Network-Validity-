"""
Space colonization algorithm for organic vascular growth.
"""

from dataclasses import dataclass
from typing import List, Optional, Set
import numpy as np
from ..core.types import Point3D, Direction3D
from ..core.network import VascularNetwork
from ..core.result import OperationResult, OperationStatus, Delta
from ..rules.constraints import BranchingConstraints


@dataclass
class SpaceColonizationParams:
    """Parameters for space colonization algorithm."""
    
    influence_radius: float = 0.015  # 15mm - radius within which tissue points attract tips
    kill_radius: float = 0.003  # 3mm - radius within which tissue points are "perfused"
    step_size: float = 0.005  # 5mm - growth step size
    min_radius: float = 0.0003  # 0.3mm - minimum vessel radius
    taper_factor: float = 0.95  # Radius reduction per generation
    vessel_type: str = "arterial"
    max_steps: int = 100  # Maximum growth steps per call
    
    preferred_direction: Optional[tuple] = None  # (x, y, z) preferred growth direction
    directional_bias: float = 0.0  # 0-1: weight for preferred direction (0=pure attraction, 1=pure directional)
    max_deviation_deg: float = 180.0  # Maximum angle deviation from preferred direction (hard constraint)
    smoothing_weight: float = 0.2  # 0-1: weight for previous direction smoothing
    
    encourage_bifurcation: bool = False  # Whether to encourage multiple children per node
    min_attractions_for_bifurcation: int = 3  # Minimum attraction points needed to consider bifurcation
    max_children_per_node: int = 2  # Maximum children to create (typically 2 for bifurcation)
    bifurcation_angle_threshold_deg: float = 40.0  # Minimum angle spread to trigger bifurcation
    bifurcation_probability: float = 0.7  # Probability of bifurcating when conditions are met
    
    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "influence_radius": self.influence_radius,
            "kill_radius": self.kill_radius,
            "step_size": self.step_size,
            "min_radius": self.min_radius,
            "taper_factor": self.taper_factor,
            "vessel_type": self.vessel_type,
            "max_steps": self.max_steps,
            "preferred_direction": self.preferred_direction,
            "directional_bias": self.directional_bias,
            "max_deviation_deg": self.max_deviation_deg,
            "smoothing_weight": self.smoothing_weight,
            "encourage_bifurcation": self.encourage_bifurcation,
            "min_attractions_for_bifurcation": self.min_attractions_for_bifurcation,
            "max_children_per_node": self.max_children_per_node,
            "bifurcation_angle_threshold_deg": self.bifurcation_angle_threshold_deg,
            "bifurcation_probability": self.bifurcation_probability,
        }
    
    @classmethod
    def from_dict(cls, d: dict) -> "SpaceColonizationParams":
        """Create from dictionary."""
        return cls(
            influence_radius=d.get("influence_radius", 0.015),
            kill_radius=d.get("kill_radius", 0.003),
            step_size=d.get("step_size", 0.005),
            min_radius=d.get("min_radius", 0.0003),
            taper_factor=d.get("taper_factor", 0.95),
            vessel_type=d.get("vessel_type", "arterial"),
            max_steps=d.get("max_steps", 100),
            preferred_direction=d.get("preferred_direction", None),
            directional_bias=d.get("directional_bias", 0.0),
            max_deviation_deg=d.get("max_deviation_deg", 180.0),
            smoothing_weight=d.get("smoothing_weight", 0.2),
            encourage_bifurcation=d.get("encourage_bifurcation", False),
            min_attractions_for_bifurcation=d.get("min_attractions_for_bifurcation", 3),
            max_children_per_node=d.get("max_children_per_node", 2),
            bifurcation_angle_threshold_deg=d.get("bifurcation_angle_threshold_deg", 40.0),
            bifurcation_probability=d.get("bifurcation_probability", 0.7),
        )


def space_colonization_step(
    network: VascularNetwork,
    tissue_points: np.ndarray,
    params: Optional[SpaceColonizationParams] = None,
    constraints: Optional[BranchingConstraints] = None,
    seed: Optional[int] = None,
) -> OperationResult:
    """
    Perform space colonization growth step.
    
    This algorithm grows vascular networks towards tissue points that need
    perfusion, creating organic space-filling patterns.
    
    Parameters
    ----------
    network : VascularNetwork
        Network to grow
    tissue_points : np.ndarray
        Array of tissue points (N, 3) that need perfusion
    params : SpaceColonizationParams, optional
        Algorithm parameters
    constraints : BranchingConstraints, optional
        Branching constraints
    seed : int, optional
        Random seed
    
    Returns
    -------
    result : OperationResult
        Result with metadata about growth progress
    
    Algorithm
    ---------
    1. For each tissue point, find nearest terminal node within influence_radius
    2. For each terminal node, compute average direction to its attracted tissue points
    3. Grow each terminal node in its attraction direction
    4. Remove tissue points within kill_radius of any node (they're "perfused")
    5. Repeat until no tissue points remain or no growth possible
    """
    if params is None:
        params = SpaceColonizationParams()
    
    if constraints is None:
        constraints = BranchingConstraints()
    
    rng = np.random.default_rng(seed) if seed is not None else network.id_gen.rng
    
    terminal_nodes = [
        node for node in network.nodes.values()
        if node.node_type in ("terminal", "inlet", "outlet") and
        node.vessel_type == params.vessel_type
    ]
    
    if not terminal_nodes:
        return OperationResult.failure(
            message=f"No terminal nodes of type {params.vessel_type} found",
            errors=["No terminal nodes"],
        )
    
    tissue_points_list = [Point3D.from_array(p) for p in tissue_points]
    active_tissue_points = set(range(len(tissue_points_list)))
    
    new_node_ids = []
    new_segment_ids = []
    warnings = []
    steps_taken = 0
    
    for step in range(params.max_steps):
        if not active_tissue_points:
            break
        
        terminal_nodes = [
            node for node in network.nodes.values()
            if node.node_type in ("terminal", "inlet", "outlet") and
            node.vessel_type == params.vessel_type
        ]
        
        attractions = {node.id: [] for node in terminal_nodes}
        
        for tp_idx in list(active_tissue_points):
            tp = tissue_points_list[tp_idx]
            
            min_dist = float('inf')
            nearest_terminal = None
            
            for node in terminal_nodes:
                dist = node.position.distance_to(tp)
                if dist < params.influence_radius and dist < min_dist:
                    min_dist = dist
                    nearest_terminal = node
            
            if nearest_terminal is not None:
                attractions[nearest_terminal.id].append(tp_idx)
        
        grown_any = False
        for node in terminal_nodes:
            if not attractions[node.id]:
                continue
            
            attracted_points = [tissue_points_list[idx] for idx in attractions[node.id]]
            avg_direction = np.zeros(3)
            
            for tp in attracted_points:
                direction = np.array([
                    tp.x - node.position.x,
                    tp.y - node.position.y,
                    tp.z - node.position.z,
                ])
                direction_norm = np.linalg.norm(direction)
                if direction_norm > 1e-10:
                    avg_direction += direction / direction_norm
            
            if np.linalg.norm(avg_direction) < 1e-10:
                continue
            
            avg_direction = avg_direction / np.linalg.norm(avg_direction)
            
            if params.preferred_direction is not None and params.directional_bias > 0:
                d_pref = np.array(params.preferred_direction)
                d_pref = d_pref / np.linalg.norm(d_pref)
                
                d_prev = None
                if "direction" in node.attributes and params.smoothing_weight > 0:
                    prev_dir = Direction3D.from_dict(node.attributes["direction"])
                    d_prev = prev_dir.to_array()
                
                # Blend attraction, preferred direction, and previous direction
                v_attr = avg_direction
                beta = params.directional_bias
                w_prev = params.smoothing_weight if d_prev is not None else 0.0
                
                if d_prev is not None:
                    blended = (1 - beta - w_prev) * v_attr + beta * d_pref + w_prev * d_prev
                else:
                    blended = (1 - beta) * v_attr + beta * d_pref
                
                blended_norm = np.linalg.norm(blended)
                if blended_norm > 1e-10:
                    blended = blended / blended_norm
                else:
                    blended = d_pref
                
                if params.max_deviation_deg < 180.0:
                    angle_to_pref = np.arccos(np.clip(np.dot(blended, d_pref), -1.0, 1.0))
                    max_angle_rad = np.radians(params.max_deviation_deg)
                    
                    if angle_to_pref > max_angle_rad:
                        axis = np.cross(blended, d_pref)
                        axis_norm = np.linalg.norm(axis)
                        
                        if axis_norm > 1e-10:
                            axis = axis / axis_norm
                            rotation_angle = angle_to_pref - max_angle_rad
                            cos_rot = np.cos(rotation_angle)
                            sin_rot = np.sin(rotation_angle)
                            
                            blended = (blended * cos_rot +
                                     np.cross(axis, blended) * sin_rot +
                                     axis * np.dot(axis, blended) * (1 - cos_rot))
                            blended = blended / np.linalg.norm(blended)
                        else:
                            blended = d_pref
                
                avg_direction = blended
            
            growth_direction = Direction3D.from_array(avg_direction)
            
            parent_radius = node.attributes.get("radius", params.min_radius * 2)
            new_radius = parent_radius * params.taper_factor
            
            if new_radius < params.min_radius:
                continue
            
            from .growth import grow_branch
            result = grow_branch(
                network,
                from_node_id=node.id,
                length=params.step_size,
                direction=growth_direction,
                target_radius=new_radius,
                constraints=constraints,
                check_collisions=True,
                seed=seed,
            )
            
            if result.is_success():
                new_node_ids.append(result.new_ids["node"])
                new_segment_ids.append(result.new_ids["segment"])
                grown_any = True
            else:
                warnings.extend(result.errors)
        
        if not grown_any:
            break
        
        steps_taken += 1
        
        for tp_idx in list(active_tissue_points):
            tp = tissue_points_list[tp_idx]
            
            for node in network.nodes.values():
                if node.position.distance_to(tp) < params.kill_radius:
                    active_tissue_points.remove(tp_idx)
                    break
    
    initial_count = len(tissue_points_list)
    perfused_count = initial_count - len(active_tissue_points)
    coverage_fraction = perfused_count / initial_count if initial_count > 0 else 0.0
    
    delta = Delta(
        created_node_ids=new_node_ids,
        created_segment_ids=new_segment_ids,
    )
    
    if new_node_ids:
        status = OperationStatus.SUCCESS if not warnings else OperationStatus.PARTIAL_SUCCESS
        message = f"Grew {len(new_node_ids)} nodes in {steps_taken} steps, {coverage_fraction:.1%} coverage"
    else:
        status = OperationStatus.WARNING
        message = "No growth occurred"
    
    return OperationResult(
        status=status,
        message=message,
        new_ids={
            "nodes": new_node_ids,
            "segments": new_segment_ids,
        },
        warnings=warnings,
        delta=delta,
        rng_state=network.id_gen.get_state(),
        metadata={
            "steps_taken": steps_taken,
            "nodes_grown": len(new_node_ids),
            "initial_tissue_points": initial_count,
            "perfused_tissue_points": perfused_count,
            "coverage_fraction": coverage_fraction,
        },
    )


def _compute_angle_spread(vectors: List[np.ndarray]) -> float:
    """
    Compute maximum pairwise angle between unit vectors.
    
    Returns angle in degrees.
    """
    if len(vectors) < 2:
        return 0.0
    
    max_angle = 0.0
    for i in range(len(vectors)):
        for j in range(i + 1, len(vectors)):
            cos_angle = np.clip(np.dot(vectors[i], vectors[j]), -1.0, 1.0)
            angle = np.degrees(np.arccos(cos_angle))
            max_angle = max(max_angle, angle)
    
    return max_angle


def _cluster_attractions_by_angle(
    attraction_vectors: List[np.ndarray],
    max_clusters: int = 2,
) -> List[List[int]]:
    """
    Cluster attraction vectors into groups using simple angle-based two-means.
    
    Returns list of cluster indices (each cluster is a list of vector indices).
    """
    if len(attraction_vectors) <= 1:
        return [[i] for i in range(len(attraction_vectors))]
    
    if max_clusters == 1 or len(attraction_vectors) <= max_clusters:
        return [[i for i in range(len(attraction_vectors))]]
    
    # Initialize centroids: pick two most separated vectors
    max_sep = -1.0
    seed_i, seed_j = 0, 1
    for i in range(len(attraction_vectors)):
        for j in range(i + 1, len(attraction_vectors)):
            sep = 1.0 - np.dot(attraction_vectors[i], attraction_vectors[j])
            if sep > max_sep:
                max_sep = sep
                seed_i, seed_j = i, j
    
    centroids = [attraction_vectors[seed_i].copy(), attraction_vectors[seed_j].copy()]
    
    # Iterative assignment (max 10 iterations)
    for iteration in range(10):
        clusters = [[] for _ in range(max_clusters)]
        
        # Assign each vector to nearest centroid
        for idx, vec in enumerate(attraction_vectors):
            best_cluster = 0
            best_sim = np.dot(vec, centroids[0])
            
            for c in range(1, max_clusters):
                sim = np.dot(vec, centroids[c])
                if sim > best_sim:
                    best_sim = sim
                    best_cluster = c
            
            clusters[best_cluster].append(idx)
        
        # Update centroids
        changed = False
        for c in range(max_clusters):
            if clusters[c]:
                new_centroid = np.mean([attraction_vectors[idx] for idx in clusters[c]], axis=0)
                new_centroid = new_centroid / np.linalg.norm(new_centroid)
                
                if np.linalg.norm(new_centroid - centroids[c]) > 1e-6:
                    changed = True
                    centroids[c] = new_centroid
        
        if not changed:
            break
    
    # Filter out empty clusters
    clusters = [c for c in clusters if c]
    
    return clusters if clusters else [[i for i in range(len(attraction_vectors))]]


def _apply_directional_blending(
    avg_direction: np.ndarray,
    node,
    params: SpaceColonizationParams,
) -> np.ndarray:
    """Apply directional constraint blending to a growth direction."""
    if params.preferred_direction is not None and params.directional_bias > 0:
        d_pref = np.array(params.preferred_direction)
        d_pref = d_pref / np.linalg.norm(d_pref)
        
        d_prev = None
        if "direction" in node.attributes and params.smoothing_weight > 0:
            prev_dir = Direction3D.from_dict(node.attributes["direction"])
            d_prev = prev_dir.to_array()
        
        v_attr = avg_direction
        beta = params.directional_bias
        w_prev = params.smoothing_weight if d_prev is not None else 0.0
        
        if d_prev is not None:
            blended = (1 - beta - w_prev) * v_attr + beta * d_pref + w_prev * d_prev
        else:
            blended = (1 - beta) * v_attr + beta * d_pref
        
        blended_norm = np.linalg.norm(blended)
        if blended_norm > 1e-10:
            blended = blended / blended_norm
        else:
            blended = d_pref
        
        if params.max_deviation_deg < 180.0:
            angle_to_pref = np.arccos(np.clip(np.dot(blended, d_pref), -1.0, 1.0))
            max_angle_rad = np.radians(params.max_deviation_deg)
            
            if angle_to_pref > max_angle_rad:
                axis = np.cross(blended, d_pref)
                axis_norm = np.linalg.norm(axis)
                
                if axis_norm > 1e-10:
                    axis = axis / axis_norm
                    rotation_angle = angle_to_pref - max_angle_rad
                    cos_rot = np.cos(rotation_angle)
                    sin_rot = np.sin(rotation_angle)
                    
                    blended = (blended * cos_rot +
                             np.cross(axis, blended) * sin_rot +
                             axis * np.dot(axis, blended) * (1 - cos_rot))
                    blended = blended / np.linalg.norm(blended)
                else:
                    blended = d_pref
        
        return blended
    
    return avg_direction
