"""
Tests for node junction metrics computation.
"""

import pytest
import numpy as np
import networkx as nx
from vascular_network.analysis.node_metrics import (
    compute_node_junction_metrics,
    compute_node_display_sizes,
)


def test_compute_node_junction_metrics_simple():
    """Test node metrics on a simple Y-junction."""
    G = nx.Graph()
    
    G.add_node(0, coord=[0, 0, 0], radius=0.003, node_type='inlet')
    G.add_node(1, coord=[0.01, 0, 0], radius=0.002, node_type='junction')
    G.add_node(2, coord=[0.02, 0.01, 0], radius=0.0015, node_type='terminal')
    G.add_node(3, coord=[0.02, -0.01, 0], radius=0.0015, node_type='terminal')
    
    G.add_edge(0, 1, radius=0.0025)
    G.add_edge(1, 2, radius=0.0015)
    G.add_edge(1, 3, radius=0.0015)
    
    metrics = compute_node_junction_metrics(G)
    
    assert metrics[0]['degree'] == 1
    assert metrics[0]['node_type'] == 'inlet'
    assert len(metrics[0]['incident_radii']) == 1
    
    assert metrics[1]['degree'] == 3
    assert metrics[1]['node_type'] == 'junction'
    assert len(metrics[1]['incident_radii']) == 3
    assert metrics[1]['effective_radius_murray3'] > 0
    assert metrics[1]['mean_radius'] > 0
    assert metrics[1]['max_radius'] > 0
    
    assert metrics[2]['degree'] == 1
    assert metrics[2]['node_type'] == 'terminal'
    assert metrics[3]['degree'] == 1
    assert metrics[3]['node_type'] == 'terminal'


def test_murray_law_aggregation():
    """Test that Murray's law aggregation is computed correctly."""
    G = nx.Graph()
    
    G.add_node(0, radius=0.003)
    G.add_node(1, radius=0.002)
    G.add_node(2, radius=0.002)
    G.add_node(3, radius=0.002)
    
    G.add_edge(0, 1, radius=0.003)
    G.add_edge(1, 2, radius=0.002)
    G.add_edge(1, 3, radius=0.002)
    
    metrics = compute_node_junction_metrics(G)
    
    expected_murray = np.power(0.003**3 + 0.002**3 + 0.002**3, 1.0/3.0)
    
    assert metrics[1]['degree'] == 3
    assert metrics[1]['effective_radius_murray3'] == pytest.approx(expected_murray, rel=0.01)


def test_compute_node_display_sizes():
    """Test display size computation."""
    G = nx.Graph()
    
    G.add_node(0, coord=[0, 0, 0], radius=0.003, node_type='inlet')
    G.add_node(1, coord=[0.01, 0, 0], radius=0.002, node_type='junction')
    G.add_node(2, coord=[0.02, 0, 0], radius=0.0015, node_type='terminal')
    
    G.add_edge(0, 1, radius=0.0025)
    G.add_edge(1, 2, radius=0.0015)
    
    sizes = compute_node_display_sizes(G, size_by='junction')
    
    assert all(s > 0 for s in sizes.values())
    
    assert sizes[1] > sizes[2]
    
    assert sizes[0] > sizes[2]


def test_display_sizes_clamping():
    """Test that display sizes are clamped to min/max."""
    G = nx.Graph()
    
    G.add_node(0, radius=0.00001)
    G.add_node(1, radius=0.1)
    
    sizes = compute_node_display_sizes(G, min_px=2.0, max_px=50.0)
    
    assert all(2.0 <= s <= 50.0 for s in sizes.values())


def test_size_by_options():
    """Test different sizing methods."""
    G = nx.Graph()
    
    G.add_node(0, radius=0.003)
    G.add_node(1, radius=0.002)
    G.add_node(2, radius=0.001)
    
    G.add_edge(0, 1)
    G.add_edge(1, 2)
    
    for size_by in ['junction', 'max_radius', 'mean_radius', 'degree']:
        sizes = compute_node_display_sizes(G, size_by=size_by)
        assert len(sizes) == 3
        assert all(s > 0 for s in sizes.values())


def test_node_type_adjustments():
    """Test that different node types get appropriate size adjustments."""
    G = nx.Graph()
    
    G.add_node(0, radius=0.002, node_type='inlet')
    G.add_node(1, radius=0.002, node_type='junction')
    G.add_node(2, radius=0.002, node_type='terminal')
    G.add_node(3, radius=0.002, node_type='outlet')
    
    G.add_edge(0, 1)
    G.add_edge(1, 2)
    G.add_edge(1, 3)
    
    sizes = compute_node_display_sizes(G, size_by='mean_radius')
    
    assert sizes[0] > sizes[2]  # inlet > terminal
    assert sizes[3] > sizes[2]  # outlet > terminal
