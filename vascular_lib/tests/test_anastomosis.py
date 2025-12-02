"""
Tests for anastomosis operations.
"""

import pytest
import numpy as np
from vascular_lib.core.network import VascularNetwork
from vascular_lib.core.types import Point3D, Direction3D
from vascular_lib.core.domain import EllipsoidDomain
from vascular_lib.ops import create_network, add_inlet, add_outlet, grow_branch
from vascular_lib.ops.anastomosis import create_anastomosis, check_tree_interactions
from vascular_lib.rules.constraints import InteractionRuleSpec


def test_create_anastomosis_basic():
    """Test basic anastomosis creation between arterial and venous nodes."""
    domain = EllipsoidDomain(center=(0, 0, 0), semi_axis_a=0.1, semi_axis_b=0.1, semi_axis_c=0.1)
    network = create_network(domain).metadata["network"]
    
    arterial_inlet = add_inlet(
        network,
        position=(0, 0, 0),
        direction=(1, 0, 0),
        radius=0.005,
        vessel_type="arterial",
    )
    arterial_id = arterial_inlet.new_ids["node"]
    
    arterial_result = grow_branch(
        network,
        from_node_id=arterial_id,
        length=0.01,
        direction=(1, 0, 0),
        target_radius=0.003,
    )
    arterial_terminal = arterial_result.new_ids["node"]
    
    venous_outlet = add_outlet(
        network,
        position=(0.015, 0, 0),
        direction=(-1, 0, 0),
        radius=0.005,
        vessel_type="venous",
    )
    venous_id = venous_outlet.new_ids["node"]
    
    venous_result = grow_branch(
        network,
        from_node_id=venous_id,
        length=0.004,
        direction=(-1, 0, 0),
        target_radius=0.003,
    )
    venous_terminal = venous_result.new_ids["node"]
    
    result = create_anastomosis(
        network,
        arterial_node_id=arterial_terminal,
        venous_node_id=venous_terminal,
    )
    
    assert result.is_success()
    assert "segment" in result.new_ids
    
    seg_id = result.new_ids["segment"]
    segment = network.segments[seg_id]
    assert segment.vessel_type == "capillary"
    assert segment.attributes.get("segment_kind") == "anastomosis"
    assert segment.attributes.get("resistance_factor") == 100.0


def test_create_anastomosis_too_long():
    """Test that anastomosis fails if nodes are too far apart."""
    domain = EllipsoidDomain(center=(0, 0, 0), semi_axis_a=0.1, semi_axis_b=0.1, semi_axis_c=0.1)
    network = create_network(domain).metadata["network"]
    
    arterial_result = add_inlet(
        network,
        position=(0, 0, 0),
        direction=(1, 0, 0),
        radius=0.005,
        vessel_type="arterial",
    )
    arterial_id = arterial_result.new_ids["node"]
    
    venous_result = add_outlet(
        network,
        position=(0.05, 0, 0),  # 50mm away - too far
        direction=(-1, 0, 0),
        radius=0.005,
        vessel_type="venous",
    )
    venous_id = venous_result.new_ids["node"]
    
    result = create_anastomosis(
        network,
        arterial_node_id=arterial_id,
        venous_node_id=venous_id,
        max_length=0.010,  # 10mm max
    )
    
    assert result.is_failure()
    assert "ANASTOMOSIS_TOO_LONG" in result.error_codes


def test_create_anastomosis_incompatible_types():
    """Test that anastomosis fails for incompatible vessel types."""
    domain = EllipsoidDomain(center=(0, 0, 0), semi_axis_a=0.1, semi_axis_b=0.1, semi_axis_c=0.1)
    network = create_network(domain).metadata["network"]
    
    arterial1 = add_inlet(
        network,
        position=(0, 0, 0),
        direction=(1, 0, 0),
        radius=0.005,
        vessel_type="arterial",
    )
    
    arterial2 = add_inlet(
        network,
        position=(0.005, 0, 0),
        direction=(1, 0, 0),
        radius=0.005,
        vessel_type="arterial",
    )
    
    result = create_anastomosis(
        network,
        arterial_node_id=arterial1.new_ids["node"],
        venous_node_id=arterial2.new_ids["node"],
    )
    
    assert result.is_failure()
    assert "INCOMPATIBLE_VESSEL_TYPES" in result.error_codes


def test_create_anastomosis_dry_run():
    """Test dry_run mode for anastomosis."""
    domain = EllipsoidDomain(center=(0, 0, 0), semi_axis_a=0.1, semi_axis_b=0.1, semi_axis_c=0.1)
    network = create_network(domain).metadata["network"]
    
    arterial = add_inlet(
        network,
        position=(0, 0, 0),
        direction=(1, 0, 0),
        radius=0.005,
        vessel_type="arterial",
    )
    
    venous = add_outlet(
        network,
        position=(0.005, 0, 0),
        direction=(-1, 0, 0),
        radius=0.005,
        vessel_type="venous",
    )
    
    initial_segment_count = len(network.segments)
    
    result = create_anastomosis(
        network,
        arterial_node_id=arterial.new_ids["node"],
        venous_node_id=venous.new_ids["node"],
        dry_run=True,
    )
    
    assert result.is_success()
    assert result.metadata["dry_run"] is True
    assert len(network.segments) == initial_segment_count  # No change


def test_check_tree_interactions():
    """Test checking interactions between arterial and venous trees."""
    domain = EllipsoidDomain(center=(0, 0, 0), semi_axis_a=0.1, semi_axis_b=0.1, semi_axis_c=0.1)
    network = create_network(domain).metadata["network"]
    
    arterial_inlet = add_inlet(
        network,
        position=(0, 0, 0),
        direction=(1, 0, 0),
        radius=0.005,
        vessel_type="arterial",
    )
    
    for i in range(3):
        grow_branch(
            network,
            from_node_id=arterial_inlet.new_ids["node"],
            length=0.005,
            direction=(1, 0, 0),
            target_radius=0.003,
        )
    
    venous_outlet = add_outlet(
        network,
        position=(0.015, 0, 0),
        direction=(-1, 0, 0),
        radius=0.005,
        vessel_type="venous",
    )
    
    for i in range(3):
        grow_branch(
            network,
            from_node_id=venous_outlet.new_ids["node"],
            length=0.005,
            direction=(-1, 0, 0),
            target_radius=0.003,
        )
    
    result = check_tree_interactions(network)
    
    assert result.is_success() or result.status.value == "warning"
    assert "violations" in result.metadata
    assert "anastomosis_candidates" in result.metadata
    assert "clearance_stats" in result.metadata


def test_check_tree_interactions_single_tree():
    """Test interaction check with only one tree type."""
    domain = EllipsoidDomain(center=(0, 0, 0), semi_axis_a=0.1, semi_axis_b=0.1, semi_axis_c=0.1)
    network = create_network(domain).metadata["network"]
    
    add_inlet(
        network,
        position=(0, 0, 0),
        direction=(1, 0, 0),
        radius=0.005,
        vessel_type="arterial",
    )
    
    result = check_tree_interactions(network)
    
    assert result.is_success()
    assert result.metadata["arterial_count"] > 0
    assert result.metadata["venous_count"] == 0
