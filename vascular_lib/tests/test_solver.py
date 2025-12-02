"""
Tests for full network flow solver.
"""

import pytest
import numpy as np
from vascular_lib.core.domain import EllipsoidDomain
from vascular_lib.core.network import VascularNetwork
from vascular_lib.ops.build import create_network, add_inlet, add_outlet
from vascular_lib.ops.growth import grow_branch, bifurcate
from vascular_lib.analysis.solver import solve_flow, check_flow_plausibility
from vascular_lib.core.types import Direction3D


def test_solve_flow_simple():
    """Test flow solver on simple linear network."""
    domain = EllipsoidDomain(semi_axis_a=0.1, semi_axis_b=0.1, semi_axis_c=0.1)
    network = create_network(domain)
    
    inlet_result = add_inlet(
        network,
        position=(-0.05, 0, 0),
        direction=Direction3D(dx=1, dy=0, dz=0),
        radius=0.005,
    )
    
    grow_result = grow_branch(
        network,
        from_node_id=inlet_result.new_ids["node"],
        length=0.08,
        direction=Direction3D(dx=1, dy=0, dz=0),
        target_radius=0.005,
    )
    
    end_node = network.nodes[grow_result.new_ids["node"]]
    end_node.node_type = "outlet"
    
    result = solve_flow(
        network,
        pin=13000.0,
        pout=2000.0,
        mu=1.0e-3,
    )
    
    assert result.is_success()
    assert 'inlet_flow' in result.metadata
    assert 'outlet_flow' in result.metadata
    
    balance_error = result.metadata['flow_balance_error']
    assert balance_error < 0.05, f"Flow balance error {balance_error} exceeds 5%"


def test_solve_flow_bifurcation():
    """Test flow solver on Y-branch."""
    domain = EllipsoidDomain(semi_axis_a=0.1, semi_axis_b=0.1, semi_axis_c=0.1)
    network = create_network(domain)
    
    inlet_result = add_inlet(
        network,
        position=(-0.05, 0, 0),
        direction=Direction3D(dx=1, dy=0, dz=0),
        radius=0.005,
    )
    
    trunk_result = grow_branch(
        network,
        from_node_id=inlet_result.new_ids["node"],
        length=0.03,
        direction=Direction3D(dx=1, dy=0, dz=0),
        target_radius=0.005,
    )
    
    bifurc_result = bifurcate(
        network,
        at_node_id=trunk_result.new_ids["node"],
        child_lengths=[0.03, 0.03],
        child_directions=[
            Direction3D(dx=0.7, dy=0.7, dz=0),
            Direction3D(dx=0.7, dy=-0.7, dz=0),
        ],
        radius_rule="murray",
    )
    
    for child_id in bifurc_result.new_ids["child_nodes"]:
        network.nodes[child_id].node_type = "outlet"
    
    result = solve_flow(network, pin=13000.0, pout=2000.0)
    
    assert result.is_success()
    
    inlet_flow = result.metadata['inlet_flow']
    outlet_flow = result.metadata['outlet_flow']
    
    assert inlet_flow > 0
    assert outlet_flow > 0


def test_flow_conservation():
    """Test that flow is conserved at junctions."""
    domain = EllipsoidDomain(semi_axis_a=0.1, semi_axis_b=0.1, semi_axis_c=0.1)
    network = create_network(domain)
    
    inlet_result = add_inlet(
        network,
        position=(-0.05, 0, 0),
        direction=Direction3D(dx=1, dy=0, dz=0),
        radius=0.005,
    )
    
    trunk_result = grow_branch(
        network,
        from_node_id=inlet_result.new_ids["node"],
        length=0.03,
        direction=Direction3D(dx=1, dy=0, dz=0),
        target_radius=0.005,
    )
    
    bifurc_result = bifurcate(
        network,
        at_node_id=trunk_result.new_ids["node"],
        child_lengths=[0.03, 0.03],
        child_directions=[
            Direction3D(dx=0.7, dy=0.7, dz=0),
            Direction3D(dx=0.7, dy=-0.7, dz=0),
        ],
        radius_rule="murray",
    )
    
    for child_id in bifurc_result.new_ids["child_nodes"]:
        network.nodes[child_id].node_type = "outlet"
    
    solve_result = solve_flow(network)
    assert solve_result.is_success()
    
    plausibility_result = check_flow_plausibility(network)
    assert plausibility_result.is_success()
    
    assert len(plausibility_result.warnings) < 5


def test_pressure_monotonicity():
    """Test that pressure decreases along flow direction."""
    domain = EllipsoidDomain(semi_axis_a=0.1, semi_axis_b=0.1, semi_axis_c=0.1)
    network = create_network(domain)
    
    inlet_result = add_inlet(
        network,
        position=(-0.05, 0, 0),
        direction=Direction3D(dx=1, dy=0, dz=0),
        radius=0.005,
    )
    
    grow_result = grow_branch(
        network,
        from_node_id=inlet_result.new_ids["node"],
        length=0.08,
        direction=Direction3D(dx=1, dy=0, dz=0),
        target_radius=0.005,
    )
    
    end_node = network.nodes[grow_result.new_ids["node"]]
    end_node.node_type = "outlet"
    
    # Solve
    solve_flow(network, pin=13000.0, pout=2000.0)
    
    inlet_node = network.nodes[inlet_result.new_ids["node"]]
    outlet_node = network.nodes[grow_result.new_ids["node"]]
    
    p_inlet = inlet_node.attributes.get('pressure', 0)
    p_outlet = outlet_node.attributes.get('pressure', 0)
    
    assert p_inlet > p_outlet, "Pressure should decrease from inlet to outlet"
