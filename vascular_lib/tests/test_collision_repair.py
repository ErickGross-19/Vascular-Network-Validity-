"""
Tests for collision repair strategies.
"""

import pytest
import numpy as np
from vascular_lib.core.domain import EllipsoidDomain
from vascular_lib.core.network import VascularNetwork
from vascular_lib.ops.build import create_network, add_inlet
from vascular_lib.ops.growth import grow_branch
from vascular_lib.ops.collision import get_collisions, avoid_collisions
from vascular_lib.core.types import Direction3D


def test_collision_detection():
    """Test that collisions are detected."""
    domain = EllipsoidDomain(semi_axis_a=0.1, semi_axis_b=0.1, semi_axis_c=0.1)
    network = create_network(domain)
    
    inlet1_result = add_inlet(
        network,
        position=(0, 0, 0),
        direction=Direction3D(dx=1, dy=0, dz=0),
        radius=0.005,
    )
    
    inlet2_result = add_inlet(
        network,
        position=(0, 0.008, 0),  # 8mm apart, but radii are 5mm each
        direction=Direction3D(dx=1, dy=0, dz=0),
        radius=0.005,
    )
    
    grow_branch(
        network,
        from_node_id=inlet1_result.new_ids["node_id"],
        length=0.03,
        direction=Direction3D(dx=1, dy=0, dz=0),
        target_radius=0.005,
    )
    
    grow_branch(
        network,
        from_node_id=inlet2_result.new_ids["node_id"],
        length=0.03,
        direction=Direction3D(dx=1, dy=0, dz=0),
        target_radius=0.005,
    )
    
    result = get_collisions(network, min_clearance=0.001)
    
    assert result.status.value == "warning"
    assert result.metadata['count'] > 0


def test_repair_by_shrink():
    """Test collision repair by shrinking."""
    domain = EllipsoidDomain(semi_axis_a=0.1, semi_axis_b=0.1, semi_axis_c=0.1)
    network = create_network(domain)
    
    inlet1_result = add_inlet(
        network,
        position=(0, 0, 0),
        direction=Direction3D(dx=1, dy=0, dz=0),
        radius=0.005,
    )
    
    inlet2_result = add_inlet(
        network,
        position=(0, 0.008, 0),
        direction=Direction3D(dx=1, dy=0, dz=0),
        radius=0.005,
    )
    
    seg1_result = grow_branch(
        network,
        from_node_id=inlet1_result.new_ids["node_id"],
        length=0.03,
        direction=Direction3D(dx=1, dy=0, dz=0),
        target_radius=0.005,
    )
    
    seg2_result = grow_branch(
        network,
        from_node_id=inlet2_result.new_ids["node_id"],
        length=0.03,
        direction=Direction3D(dx=1, dy=0, dz=0),
        target_radius=0.005,
    )
    
    seg1 = network.segments[seg1_result.new_ids["segment_id"]]
    initial_radius = seg1.geometry.radius_start
    
    repair_result = avoid_collisions(network, min_clearance=0.001, repair_strategy="shrink")
    
    final_radius = seg1.geometry.radius_start
    assert final_radius < initial_radius


def test_repair_by_terminate():
    """Test collision repair by terminating."""
    domain = EllipsoidDomain(semi_axis_a=0.1, semi_axis_b=0.1, semi_axis_c=0.1)
    network = create_network(domain)
    
    inlet1_result = add_inlet(
        network,
        position=(0, 0, 0),
        direction=Direction3D(dx=1, dy=0, dz=0),
        radius=0.005,
    )
    
    inlet2_result = add_inlet(
        network,
        position=(0, 0.008, 0),
        direction=Direction3D(dx=1, dy=0, dz=0),
        radius=0.005,
    )
    
    seg1_result = grow_branch(
        network,
        from_node_id=inlet1_result.new_ids["node_id"],
        length=0.03,
        direction=Direction3D(dx=1, dy=0, dz=0),
        target_radius=0.005,
    )
    
    seg2_result = grow_branch(
        network,
        from_node_id=inlet2_result.new_ids["node_id"],
        length=0.03,
        direction=Direction3D(dx=1, dy=0, dz=0),
        target_radius=0.005,
    )
    
    node1 = network.nodes[seg1_result.new_ids["node_id"]]
    initial_type = node1.node_type
    
    repair_result = avoid_collisions(network, min_clearance=0.001, repair_strategy="terminate")
    
    assert repair_result.is_success()
    
    terminal_count = sum(1 for node in network.nodes.values() if node.node_type == "terminal")
    assert terminal_count > 0


def test_no_collisions():
    """Test that well-separated branches don't trigger collisions."""
    domain = EllipsoidDomain(semi_axis_a=0.1, semi_axis_b=0.1, semi_axis_c=0.1)
    network = create_network(domain)
    
    inlet1_result = add_inlet(
        network,
        position=(0, 0, 0),
        direction=Direction3D(dx=1, dy=0, dz=0),
        radius=0.003,
    )
    
    inlet2_result = add_inlet(
        network,
        position=(0, 0.05, 0),  # 50mm apart
        direction=Direction3D(dx=1, dy=0, dz=0),
        radius=0.003,
    )
    
    grow_branch(
        network,
        from_node_id=inlet1_result.new_ids["node_id"],
        length=0.03,
        direction=Direction3D(dx=1, dy=0, dz=0),
        target_radius=0.003,
    )
    
    grow_branch(
        network,
        from_node_id=inlet2_result.new_ids["node_id"],
        length=0.03,
        direction=Direction3D(dx=1, dy=0, dz=0),
        target_radius=0.003,
    )
    
    result = get_collisions(network, min_clearance=0.001)
    
    assert result.is_success()
    assert result.metadata['count'] == 0
