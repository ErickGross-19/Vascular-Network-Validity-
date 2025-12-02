"""
Tests for perfusion analysis.
"""

import pytest
import numpy as np
from vascular_lib.core.network import VascularNetwork
from vascular_lib.core.domain import EllipsoidDomain
from vascular_lib.ops import create_network, add_inlet, add_outlet, grow_branch
from vascular_lib.analysis.perfusion import compute_perfusion_metrics, suggest_anastomosis_locations
from vascular_lib.rules.constraints import InteractionRuleSpec


def test_compute_perfusion_metrics_dual_tree():
    """Test perfusion metrics with both arterial and venous trees."""
    domain = EllipsoidDomain(center=(0, 0, 0), semi_axis_a=0.05, semi_axis_b=0.05, semi_axis_c=0.05)
    network = create_network(domain).metadata["network"]
    
    arterial_inlet = add_inlet(
        network,
        position=(-0.02, 0, 0),
        direction=(1, 0, 0),
        radius=0.005,
        vessel_type="arterial",
    )
    
    grow_branch(
        network,
        from_node_id=arterial_inlet.new_ids["node"],
        length=0.015,
        direction=(1, 0, 0),
        target_radius=0.003,
    )
    
    venous_outlet = add_outlet(
        network,
        position=(0.02, 0, 0),
        direction=(-1, 0, 0),
        radius=0.005,
        vessel_type="venous",
    )
    
    grow_branch(
        network,
        from_node_id=venous_outlet.new_ids["node"],
        length=0.015,
        direction=(-1, 0, 0),
        target_radius=0.003,
    )
    
    tissue_points = np.array([
        [0, 0, 0],  # Center - should be well perfused
        [0, 0.03, 0],  # Far from vessels - poorly perfused
        [-0.01, 0, 0],  # Near arterial
        [0.01, 0, 0],  # Near venous
    ])
    
    metrics = compute_perfusion_metrics(network, tissue_points)
    
    assert "perfusion_scores" in metrics
    assert "arterial_distances" in metrics
    assert "venous_distances" in metrics
    assert "well_perfused_fraction" in metrics
    assert "under_perfused_regions" in metrics
    
    assert len(metrics["perfusion_scores"]) == len(tissue_points)
    assert 0 <= metrics["well_perfused_fraction"] <= 1.0
    
    assert metrics["perfusion_scores"][0] > metrics["perfusion_scores"][1]


def test_compute_perfusion_metrics_single_tree():
    """Test perfusion metrics with only one tree type."""
    domain = EllipsoidDomain(center=(0, 0, 0), semi_axis_a=0.05, semi_axis_b=0.05, semi_axis_c=0.05)
    network = create_network(domain).metadata["network"]
    
    arterial_inlet = add_inlet(
        network,
        position=(0, 0, 0),
        direction=(1, 0, 0),
        radius=0.005,
        vessel_type="arterial",
    )
    
    tissue_points = np.array([[0.01, 0, 0]])
    
    metrics = compute_perfusion_metrics(network, tissue_points)
    
    assert metrics["arterial_node_count"] > 0
    assert len(metrics["perfusion_scores"]) == 1


def test_compute_perfusion_metrics_with_weights():
    """Test perfusion metrics with custom weights."""
    domain = EllipsoidDomain(center=(0, 0, 0), semi_axis_a=0.05, semi_axis_b=0.05, semi_axis_c=0.05)
    network = create_network(domain).metadata["network"]
    
    add_inlet(
        network,
        position=(-0.01, 0, 0),
        direction=(1, 0, 0),
        radius=0.005,
        vessel_type="arterial",
    )
    
    add_outlet(
        network,
        position=(0.01, 0, 0),
        direction=(-1, 0, 0),
        radius=0.005,
        vessel_type="venous",
    )
    
    tissue_points = np.array([[0, 0, 0]])
    
    metrics1 = compute_perfusion_metrics(network, tissue_points, weights=(1.0, 1.0))
    
    metrics2 = compute_perfusion_metrics(network, tissue_points, weights=(2.0, 1.0))
    
    assert metrics1["perfusion_scores"][0] != metrics2["perfusion_scores"][0]


def test_compute_perfusion_metrics_with_distance_cap():
    """Test perfusion metrics with distance cap."""
    domain = EllipsoidDomain(center=(0, 0, 0), semi_axis_a=0.1, semi_axis_b=0.1, semi_axis_c=0.1)
    network = create_network(domain).metadata["network"]
    
    add_inlet(
        network,
        position=(0, 0, 0),
        direction=(1, 0, 0),
        radius=0.005,
        vessel_type="arterial",
    )
    
    add_outlet(
        network,
        position=(0.05, 0, 0),
        direction=(-1, 0, 0),
        radius=0.005,
        vessel_type="venous",
    )
    
    tissue_points = np.array([[0, 0.08, 0]])
    
    metrics1 = compute_perfusion_metrics(network, tissue_points)
    
    metrics2 = compute_perfusion_metrics(network, tissue_points, distance_cap=0.01)
    
    assert metrics2["arterial_distances"][0] <= 0.01
    assert metrics2["venous_distances"][0] <= 0.01


def test_suggest_anastomosis_locations():
    """Test suggesting anastomosis locations based on perfusion."""
    domain = EllipsoidDomain(center=(0, 0, 0), semi_axis_a=0.05, semi_axis_b=0.05, semi_axis_c=0.05)
    network = create_network(domain).metadata["network"]
    
    arterial_inlet = add_inlet(
        network,
        position=(-0.02, 0, 0),
        direction=(1, 0, 0),
        radius=0.005,
        vessel_type="arterial",
    )
    
    arterial_result = grow_branch(
        network,
        from_node_id=arterial_inlet.new_ids["node"],
        length=0.01,
        direction=(1, 0, 0),
        target_radius=0.003,
    )
    arterial_terminal = arterial_result.new_ids["node"]
    
    venous_outlet = add_outlet(
        network,
        position=(0.02, 0, 0),
        direction=(-1, 0, 0),
        radius=0.005,
        vessel_type="venous",
    )
    
    venous_result = grow_branch(
        network,
        from_node_id=venous_outlet.new_ids["node"],
        length=0.01,
        direction=(-1, 0, 0),
        target_radius=0.003,
    )
    venous_terminal = venous_result.new_ids["node"]
    
    tissue_points = np.array([
        [0, 0, 0],  # Center - under-perfused
        [0, 0.01, 0],
        [0, -0.01, 0],
    ])
    
    perfusion_metrics = compute_perfusion_metrics(network, tissue_points)
    
    candidates = suggest_anastomosis_locations(
        network,
        perfusion_metrics,
        k=3,
    )
    
    assert isinstance(candidates, list)
    
    if len(candidates) > 0:
        candidate = candidates[0]
        assert "arterial_node" in candidate
        assert "venous_node" in candidate
        assert "distance" in candidate
        assert "score" in candidate
        assert "reason" in candidate


def test_suggest_anastomosis_no_under_perfused():
    """Test suggesting anastomoses when perfusion is good."""
    domain = EllipsoidDomain(center=(0, 0, 0), semi_axis_a=0.02, semi_axis_b=0.02, semi_axis_c=0.02)
    network = create_network(domain).metadata["network"]
    
    add_inlet(
        network,
        position=(0, 0, 0),
        direction=(1, 0, 0),
        radius=0.005,
        vessel_type="arterial",
    )
    
    add_outlet(
        network,
        position=(0.005, 0, 0),
        direction=(-1, 0, 0),
        radius=0.005,
        vessel_type="venous",
    )
    
    tissue_points = np.array([[0.0025, 0, 0]])
    
    perfusion_metrics = compute_perfusion_metrics(network, tissue_points)
    
    assert perfusion_metrics["well_perfused_fraction"] > 0.5
    
    candidates = suggest_anastomosis_locations(network, perfusion_metrics, k=5)
    assert isinstance(candidates, list)
