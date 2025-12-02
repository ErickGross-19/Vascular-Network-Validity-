"""
Tests for degradation rules.
"""

import pytest
from vascular_lib.core.network import VascularNetwork
from vascular_lib.core.domain import EllipsoidDomain
from vascular_lib.ops import create_network, add_inlet, bifurcate
from vascular_lib.rules.constraints import DegradationRuleSpec, BranchingConstraints


def test_degradation_rule_exponential():
    """Test exponential degradation model."""
    rule = DegradationRuleSpec.exponential(factor=0.8, min_radius=0.0001)
    
    parent_radius = 0.005
    
    r1 = rule.apply_degradation(parent_radius, 1)
    assert r1 == pytest.approx(parent_radius * 0.8)
    
    r2 = rule.apply_degradation(parent_radius, 2)
    assert r2 == pytest.approx(parent_radius * (0.8 ** 2))
    
    r10 = rule.apply_degradation(parent_radius, 10)
    assert r10 >= rule.min_terminal_radius


def test_degradation_rule_linear():
    """Test linear degradation model."""
    rule = DegradationRuleSpec.linear(factor=0.9, min_radius=0.0001)
    
    parent_radius = 0.005
    
    r1 = rule.apply_degradation(parent_radius, 1)
    expected = parent_radius * (1.0 - (1.0 - 0.9) * 1)
    assert r1 == pytest.approx(expected)
    
    r20 = rule.apply_degradation(parent_radius, 20)
    assert r20 >= rule.min_terminal_radius


def test_degradation_rule_generation_based():
    """Test generation-based degradation model."""
    rule = DegradationRuleSpec.generation_based(factor=0.85, max_gen=5)
    
    parent_radius = 0.005
    
    r1 = rule.apply_degradation(parent_radius, 1)
    assert r1 == pytest.approx(parent_radius * 0.85)
    
    r2 = rule.apply_degradation(parent_radius, 2)
    assert r2 == pytest.approx(parent_radius * 0.85)


def test_degradation_termination_radius():
    """Test termination based on minimum radius."""
    rule = DegradationRuleSpec.exponential(factor=0.8, min_radius=0.0005)
    
    should_term, reason = rule.should_terminate(0.003, 1)
    assert not should_term
    
    should_term, reason = rule.should_terminate(0.0005, 5)
    assert should_term
    assert "radius" in reason.lower()


def test_degradation_termination_generation():
    """Test termination based on maximum generation."""
    rule = DegradationRuleSpec.exponential(factor=0.8, min_radius=0.0001)
    rule.max_generation = 10
    
    should_term, reason = rule.should_terminate(0.003, 5)
    assert not should_term
    
    should_term, reason = rule.should_terminate(0.003, 10)
    assert should_term
    assert "generation" in reason.lower()


def test_bifurcate_with_degradation():
    """Test bifurcation with degradation rules."""
    domain = EllipsoidDomain(semi_axis_a=0.1, semi_axis_b=0.1, semi_axis_c=0.1)
    network = create_network(domain)
    
    inlet_result = add_inlet(
        network,
        position=(0, 0, 0),
        direction=(1, 0, 0),
        radius=0.005,
    )
    inlet_id = inlet_result.new_ids["node"]
    
    degradation_rule = DegradationRuleSpec.exponential(factor=0.85, min_radius=0.0001)
    
    result = bifurcate(
        network,
        at_node_id=inlet_id,
        child_lengths=(0.01, 0.01),
        angle_deg=45.0,
        degradation_rule=degradation_rule,
    )
    
    assert result.is_success()
    
    child_nodes = result.new_ids["nodes"]
    for child_id in child_nodes:
        child_radius = network.nodes[child_id].attributes["radius"]
        assert child_radius < 0.005


def test_bifurcate_blocked_by_degradation():
    """Test that bifurcation is blocked when degradation rules prevent it."""
    domain = EllipsoidDomain(semi_axis_a=0.1, semi_axis_b=0.1, semi_axis_c=0.1)
    network = create_network(domain)
    
    inlet_result = add_inlet(
        network,
        position=(0, 0, 0),
        direction=(1, 0, 0),
        radius=0.0002,  # Very small
    )
    inlet_id = inlet_result.new_ids["node"]
    
    degradation_rule = DegradationRuleSpec.exponential(factor=0.5, min_radius=0.0002)
    
    result = bifurcate(
        network,
        at_node_id=inlet_id,
        child_lengths=(0.01, 0.01),
        angle_deg=45.0,
        degradation_rule=degradation_rule,
    )
    
    assert result.is_failure()
    assert any(code in result.error_codes for code in ["BELOW_MIN_TERMINAL_RADIUS", "RADIUS_TOO_SMALL"])


def test_degradation_serialization():
    """Test serialization of degradation rules."""
    rule = DegradationRuleSpec.exponential(factor=0.85, min_radius=0.0001)
    
    d = rule.to_dict()
    assert d["model"] == "exponential"
    assert d["degradation_factor"] == 0.85
    assert d["min_terminal_radius"] == 0.0001
    
    rule2 = DegradationRuleSpec.from_dict(d)
    assert rule2.model == rule.model
    assert rule2.degradation_factor == rule.degradation_factor
    assert rule2.min_terminal_radius == rule.min_terminal_radius
