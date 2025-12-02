"""
Tests for deterministic behavior with seeds.
"""

import pytest
from vascular_lib.core.domain import EllipsoidDomain
from vascular_lib.ops.build import create_network, add_inlet
from vascular_lib.ops.space_colonization import space_colonization_step, SpaceColonizationParams
from vascular_lib.core.types import Direction3D
import numpy as np


def test_deterministic_ids():
    """Test that same seed produces same IDs."""
    domain = EllipsoidDomain(semi_axis_a=0.1, semi_axis_b=0.1, semi_axis_c=0.1)
    
    network1 = create_network(domain, seed=42)
    network2 = create_network(domain, seed=42)
    
    result1 = add_inlet(
        network1,
        position=(0, 0, 0),
        direction=Direction3D(x=1, y=0, z=0),
        radius=0.005,
    )
    
    result2 = add_inlet(
        network2,
        position=(0, 0, 0),
        direction=Direction3D(x=1, y=0, z=0),
        radius=0.005,
    )
    
    assert result1.new_ids["node_id"] == result2.new_ids["node_id"]


def test_space_colonization_determinism():
    """Test that space colonization is deterministic with same seed."""
    domain = EllipsoidDomain(semi_axis_a=0.1, semi_axis_b=0.1, semi_axis_c=0.1)
    
    np.random.seed(123)
    tissue_points = np.random.uniform(-0.08, 0.08, size=(100, 3))
    
    network1 = create_network(domain, seed=42)
    inlet1 = add_inlet(
        network1,
        position=(-0.08, 0, 0),
        direction=Direction3D(x=1, y=0, z=0),
        radius=0.005,
        vessel_type="arterial",
    )
    
    params = SpaceColonizationParams(
        vessel_type="arterial",
        influence_radius=0.02,
        kill_radius=0.005,
        step_size=0.01,
        max_steps=5,
    )
    
    result1 = space_colonization_step(network1, tissue_points, params, seed=42)
    
    network2 = create_network(domain, seed=42)
    inlet2 = add_inlet(
        network2,
        position=(-0.08, 0, 0),
        direction=Direction3D(x=1, y=0, z=0),
        radius=0.005,
        vessel_type="arterial",
    )
    
    result2 = space_colonization_step(network2, tissue_points, params, seed=42)
    
    assert len(network1.nodes) == len(network2.nodes)
    assert len(network1.segments) == len(network2.segments)


def test_serialization_roundtrip():
    """Test that network can be serialized and deserialized."""
    domain = EllipsoidDomain(semi_axis_a=0.1, semi_axis_b=0.1, semi_axis_c=0.1)
    network1 = create_network(domain, metadata={"name": "test", "version": "1.0"})
    
    add_inlet(
        network1,
        position=(0, 0, 0),
        direction=Direction3D(x=1, y=0, z=0),
        radius=0.005,
    )
    
    network_dict = network1.to_dict()
    
    from vascular_lib.core.network import VascularNetwork
    network2 = VascularNetwork.from_dict(network_dict)
    
    assert len(network2.nodes) == len(network1.nodes)
    assert len(network2.segments) == len(network1.segments)
    assert network2.metadata == network1.metadata
    
    for node_id in network1.nodes:
        pos1 = network1.nodes[node_id].position
        pos2 = network2.nodes[node_id].position
        assert pos1.x == pos2.x
        assert pos1.y == pos2.y
        assert pos1.z == pos2.z
