"""
Tests for per-component flow measurements.
"""

import pytest
from vascular_lib.core.domain import EllipsoidDomain
from vascular_lib.ops import create_network, add_inlet, add_outlet, bifurcate
from vascular_lib.analysis import solve_flow, compute_component_flows


def test_compute_component_flows_dual_tree():
    """Test per-component flow measurements for dual-tree network."""
    domain = EllipsoidDomain(semi_axis_a=0.05, semi_axis_b=0.05, semi_axis_c=0.05)
    network = create_network(domain)
    
    add_inlet(
        network,
        position=(-0.02, 0, 0),
        direction=(1, 0, 0),
        radius=0.003,
        vessel_type="arterial",
    )
    arterial_root = list(network.nodes.values())[0].id
    bifurcate(network, arterial_root, angle_deg=30, length=0.01)
    
    add_outlet(
        network,
        position=(0.02, 0, 0),
        direction=(-1, 0, 0),
        radius=0.003,
        vessel_type="venous",
    )
    venous_root = list(network.nodes.values())[-1].id
    bifurcate(network, venous_root, angle1=30, angle2=-30, length1=0.01, length2=0.01)
    
    result = solve_flow(network)
    assert result.status == "success"
    
    component_flows = result.metadata["component_flows"]
    assert component_flows["num_components"] == 2
    
    components = component_flows["components"]
    assert len(components) == 2
    
    arterial_comp = next((c for c in components if c["vessel_type"] == "arterial"), None)
    venous_comp = next((c for c in components if c["vessel_type"] == "venous"), None)
    
    assert arterial_comp is not None
    assert venous_comp is not None
    
    assert arterial_comp["root_node_type"] == "inlet"
    assert arterial_comp["total_flow"] > 0
    assert arterial_comp["num_nodes"] > 1
    assert arterial_comp["num_terminals"] == 2  # Two branches from bifurcation
    assert arterial_comp["pressure_drop"] > 0
    
    assert venous_comp["root_node_type"] == "outlet"
    assert venous_comp["total_flow"] > 0
    assert venous_comp["num_nodes"] > 1
    assert venous_comp["num_terminals"] == 2
    assert venous_comp["pressure_drop"] > 0


def test_compute_component_flows_single_tree():
    """Test per-component flow measurements for single-tree network."""
    domain = EllipsoidDomain(semi_axis_a=0.05, semi_axis_b=0.05, semi_axis_c=0.05)
    network = create_network(domain)
    
    add_inlet(
        network,
        position=(-0.02, 0, 0),
        direction=(1, 0, 0),
        radius=0.003,
        vessel_type="arterial",
    )
    root_id = list(network.nodes.values())[0].id
    
    add_outlet(
        network,
        position=(0.02, 0, 0),
        direction=(-1, 0, 0),
        radius=0.003,
        vessel_type="arterial",
    )
    
    result = solve_flow(network)
    assert result.status == "success"
    
    component_flows = result.metadata["component_flows"]
    assert component_flows["num_components"] == 1
    
    components = component_flows["components"]
    assert len(components) == 1
    
    comp = components[0]
    assert comp["vessel_type"] == "arterial"
    assert comp["root_node_type"] == "inlet"
    assert comp["total_flow"] > 0
    assert comp["pressure_drop"] > 0


def test_compute_component_flows_no_flow():
    """Test compute_component_flows on network without flow solution."""
    domain = EllipsoidDomain(semi_axis_a=0.05, semi_axis_b=0.05, semi_axis_c=0.05)
    network = create_network(domain)
    
    add_inlet(network, position=(-0.02, 0, 0), direction=(1, 0, 0), radius=0.003)
    
    component_flows = compute_component_flows(network)
    
    assert component_flows["num_components"] == 0
    assert "warning" in component_flows
