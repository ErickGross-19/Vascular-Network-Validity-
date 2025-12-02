"""
Configuration dataclasses for liver vascular network generation.
"""

from dataclasses import dataclass, field
from typing import Tuple
import numpy as np


@dataclass
class LiverGeometryConfig:
    """Configuration for liver domain geometry."""
    
    semi_axis_a: float = 0.12  # x-axis (left-right)
    semi_axis_b: float = 0.10  # y-axis (anterior-posterior)
    semi_axis_c: float = 0.08  # z-axis (superior-inferior)
    
    arterial_root_position: Tuple[float, float, float] = (-0.8, 0.0, 0.0)  # Left side
    venous_root_position: Tuple[float, float, float] = (0.8, 0.0, 0.0)  # Right side
    
    meeting_shell_center_x: float = 0.0
    meeting_shell_thickness: float = 0.02  # meters


@dataclass
class MurrayLawConfig:
    """Configuration for Murray's law and radius scaling."""
    
    gamma: float = 3.0
    
    arterial_root_radius: float = 0.005  # 5 mm
    venous_root_radius: float = 0.006  # 6 mm (slightly larger)
    
    min_radius: float = 0.0003  # 0.3 mm
    
    split_ratio_mean: float = 0.6  # Asymmetric splits
    split_ratio_std: float = 0.1


@dataclass
class BranchingConfig:
    """Configuration for branching behavior."""
    
    max_branch_order: int = 12
    
    base_branching_prob: float = 0.3
    prob_exponent: float = 1.5
    
    branch_angle_mean: float = 50.0  # Degrees from parent
    branch_angle_std: float = 15.0
    
    
    max_curvature_per_step: float = 15.0
    
    step_size_factor: float = 2.0


@dataclass
class GrowthConfig:
    """Configuration for tree growth algorithm."""
    
    arterial_first: bool = True  # If False, interleave growth
    
    max_segments_per_tree: int = 5000
    
    collision_margin: float = 0.0005  # 0.5 mm safety margin
    soft_collision_factor: float = 1.5  # Repulsion starts at 1.5× (r1+r2)
    
    min_distance_to_boundary: float = 0.002  # 2 mm from liver surface
    
    parent_direction_weight: float = 0.7
    center_attraction_weight: float = 0.2
    random_perturbation_weight: float = 0.1
    
    direction_noise_std: float = 10.0


@dataclass
class LiverVascularConfig:
    """Complete configuration for liver vascular network generation."""
    
    geometry: LiverGeometryConfig = field(default_factory=LiverGeometryConfig)
    murray: MurrayLawConfig = field(default_factory=MurrayLawConfig)
    branching: BranchingConfig = field(default_factory=BranchingConfig)
    growth: GrowthConfig = field(default_factory=GrowthConfig)
    
    random_seed: int = 42
    
    schema_version: str = "1.0"
    
    def to_dict(self):
        """Convert config to dictionary for serialization."""
        return {
            "schema_version": self.schema_version,
            "random_seed": self.random_seed,
            "geometry": {
                "semi_axis_a": self.geometry.semi_axis_a,
                "semi_axis_b": self.geometry.semi_axis_b,
                "semi_axis_c": self.geometry.semi_axis_c,
                "arterial_root_position": self.geometry.arterial_root_position,
                "venous_root_position": self.geometry.venous_root_position,
                "meeting_shell_center_x": self.geometry.meeting_shell_center_x,
                "meeting_shell_thickness": self.geometry.meeting_shell_thickness,
            },
            "murray": {
                "gamma": self.murray.gamma,
                "arterial_root_radius": self.murray.arterial_root_radius,
                "venous_root_radius": self.murray.venous_root_radius,
                "min_radius": self.murray.min_radius,
                "split_ratio_mean": self.murray.split_ratio_mean,
                "split_ratio_std": self.murray.split_ratio_std,
            },
            "branching": {
                "max_branch_order": self.branching.max_branch_order,
                "base_branching_prob": self.branching.base_branching_prob,
                "prob_exponent": self.branching.prob_exponent,
                "branch_angle_mean": self.branching.branch_angle_mean,
                "branch_angle_std": self.branching.branch_angle_std,
                "max_curvature_per_step": self.branching.max_curvature_per_step,
                "step_size_factor": self.branching.step_size_factor,
            },
            "growth": {
                "arterial_first": self.growth.arterial_first,
                "max_segments_per_tree": self.growth.max_segments_per_tree,
                "collision_margin": self.growth.collision_margin,
                "soft_collision_factor": self.growth.soft_collision_factor,
                "min_distance_to_boundary": self.growth.min_distance_to_boundary,
                "parent_direction_weight": self.growth.parent_direction_weight,
                "center_attraction_weight": self.growth.center_attraction_weight,
                "random_perturbation_weight": self.growth.random_perturbation_weight,
                "direction_noise_std": self.growth.direction_noise_std,
            },
        }
