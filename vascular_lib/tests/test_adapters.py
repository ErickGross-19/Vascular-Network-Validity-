"""
Tests for adapters (NetworkX, mesh, report).
"""

import pytest
import numpy as np
from vascular_lib.core.domain import EllipsoidDomain
from vascular_lib.core.network import VascularNetwork
from vascular_lib.ops.build import create_network, add_inlet, add_outlet
from vascular_lib.ops.growth import grow_branch
from vascular_lib.adapters.networkx_adapter import to_networkx_graph, from_networkx_graph
from vascular_lib.adapters.mesh_adapter import to_trimesh
from vascular_lib.core.types import Direction3D


def test_to_networkx_graph():
    """Test conversion to NetworkX graph."""
    domain = EllipsoidDomain(semi_axis_a=0.1, semi_axis_b=0.1, semi_axis_c=0.1)
    network = create_network(domain, metadata={"name": "test"})
    
    inlet_result = add_inlet(
        network,
        position=(0, 0, 0),
        direction=Direction3D(dx=1, dy=0, dz=0),
        radius=0.005,
        vessel_type="arterial",
    )
    
    outlet_result = add_outlet(
        network,
        position=(0.08, 0, 0),
        direction=Direction3D(dx=1, dy=0, dz=0),
        radius=0.006,
        vessel_type="venous",
    )
    
    grow_result = grow_branch(
        network,
        from_node_id=inlet_result.new_ids["node"],
        length=0.03,
        direction=Direction3D(dx=1, dy=0, dz=0),
        target_radius=0.004,
    )
    
    G, node_id_map = to_networkx_graph(network)
    
    assert G.number_of_nodes() == 3
    assert G.number_of_edges() == 1
    
    for nx_id in G.nodes():
        assert 'coord' in G.nodes[nx_id]
        assert 'radius' in G.nodes[nx_id]
        assert len(G.nodes[nx_id]['coord']) == 3


def test_networkx_roundtrip():
    """Test NetworkX conversion round-trip."""
    domain = EllipsoidDomain(semi_axis_a=0.1, semi_axis_b=0.1, semi_axis_c=0.1)
    network1 = create_network(domain)
    
    add_inlet(network1, position=(0, 0, 0), direction=Direction3D(dx=1, dy=0, dz=0), radius=0.005)
    add_outlet(network1, position=(0.08, 0, 0), direction=Direction3D(dx=1, dy=0, dz=0), radius=0.006)
    
    G, node_id_map = to_networkx_graph(network1)
    network2 = from_networkx_graph(G, domain, node_id_map)
    
    assert len(network2.nodes) == len(network1.nodes)
    assert len(network2.segments) == len(network1.segments)


def test_to_trimesh_fast():
    """Test fast mesh export."""
    domain = EllipsoidDomain(semi_axis_a=0.1, semi_axis_b=0.1, semi_axis_c=0.1)
    network = create_network(domain)
    
    inlet_result = add_inlet(
        network,
        position=(0, 0, 0),
        direction=Direction3D(dx=1, dy=0, dz=0),
        radius=0.005,
    )
    
    grow_branch(
        network,
        from_node_id=inlet_result.new_ids["node"],
        length=0.03,
        direction=Direction3D(dx=1, dy=0, dz=0),
        target_radius=0.004,
    )
    
    result = to_trimesh(network, mode="fast", radial_resolution=8)
    
    assert result.is_success()
    assert 'mesh' in result.metadata
    
    mesh = result.metadata['mesh']
    assert mesh.vertices.shape[0] > 0
    assert mesh.faces.shape[0] > 0


def test_serialization_json_safe():
    """Test that all results are JSON-serializable."""
    import json
    
    domain = EllipsoidDomain(semi_axis_a=0.1, semi_axis_b=0.1, semi_axis_c=0.1)
    network = create_network(domain)
    
    inlet_result = add_inlet(
        network,
        position=(0, 0, 0),
        direction=Direction3D(dx=1, dy=0, dz=0),
        radius=0.005,
    )
    
    result_dict = inlet_result.to_dict()
    json_str = json.dumps(result_dict)
    
    assert len(json_str) > 0
    
    network_dict = network.to_dict()
    json_str = json.dumps(network_dict)
    
    assert len(json_str) > 0
