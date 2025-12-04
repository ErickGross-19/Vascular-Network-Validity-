"""
Tests for per-component flow measurements.
"""

import pytest
from vascular_lib.core.domain import EllipsoidDomain
from vascular_lib.ops import create_network, add_inlet, add_outlet, bifurcate, grow_branch, create_anastomosis
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
    bifurcate(network, arterial_root, child_lengths=(0.01, 0.01), angle_deg=30)
    
    add_outlet(
        network,
        position=(0.02, 0, 0),
        direction=(-1, 0, 0),
        radius=0.003,
        vessel_type="venous",
    )
    venous_root = list(network.nodes.values())[-1].id
    bifurcate(network, venous_root, child_lengths=(0.01, 0.01), angle_deg=30)
    
    arterial_terminals = [n.id for n in network.nodes.values() if n.vessel_type == "arterial" and n.node_type == "terminal"]
    venous_terminals = [n.id for n in network.nodes.values() if n.vessel_type == "venous" and n.node_type == "terminal"]
    create_anastomosis(network, arterial_terminals[0], venous_terminals[0])
    
    result = solve_flow(network)
    assert result.status.value == "success"
    
    component_flows = result.metadata["component_flows"]
    assert component_flows["num_components"] == 2
    
    components = component_flows["components"]
    assert len(components) == 2
    
    arterial_comp = next((c for c in components if c["vessel_type"] == "arterial"), None)
    venous_comp = next((c for c in components if c["vessel_type"] == "venous"), None)
    
    assert arterial_comp is not None
    assert venous_comp is not None
    
    assert arterial_comp["root_node_type"] == "inlet"
    assert arterial_comp["total_flow"] >= 0
    assert arterial_comp["num_nodes"] > 1
    assert arterial_comp["num_terminals"] == 2
    
    assert venous_comp["root_node_type"] == "outlet"
    assert venous_comp["total_flow"] >= 0
    assert venous_comp["num_nodes"] > 1
    assert venous_comp["num_terminals"] == 2


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
    inlet_id = list(network.nodes.values())[0].id
    
    grow_branch(network, from_node_id=inlet_id, direction=(1, 0, 0), length=0.04)
    terminal_id = [n.id for n in network.nodes.values() if n.node_type == "terminal"][0]
    
    add_outlet(
        network,
        position=(0.02, 0, 0),
        direction=(-1, 0, 0),
        radius=0.003,
        vessel_type="arterial",
    )
    
    result = solve_flow(network)
    assert result.status.value == "success"
    
    component_flows = result.metadata["component_flows"]
    
    components = component_flows["components"]
    assert len(components) >= 1
    
    comp = components[0]
    assert comp["vessel_type"] == "arterial"
    assert comp["root_node_type"] == "inlet"
    assert comp["total_flow"] >= 0


def test_compute_component_flows_no_flow():
    """Test compute_component_flows on network without flow solution."""
    domain = EllipsoidDomain(semi_axis_a=0.05, semi_axis_b=0.05, semi_axis_c=0.05)
    network = create_network(domain)
    
    add_inlet(network, position=(-0.02, 0, 0), direction=(1, 0, 0), radius=0.003)
    
    component_flows = compute_component_flows(network)
    
    assert component_flows["num_components"] == 0
    assert "warning" in component_flows
