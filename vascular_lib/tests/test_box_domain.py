"""
Tests for BoxDomain.
"""

import pytest
import numpy as np
from vascular_lib.core.domain import BoxDomain
from vascular_lib.core.types import Point3D


def test_box_domain_creation():
    """Test BoxDomain creation."""
    box = BoxDomain(
        x_min=-0.05, x_max=0.05,
        y_min=-0.03, y_max=0.03,
        z_min=-0.02, z_max=0.02,
    )
    
    assert box.x_min == -0.05
    assert box.x_max == 0.05
    assert box.y_min == -0.03
    assert box.y_max == 0.03
    assert box.z_min == -0.02
    assert box.z_max == 0.02


def test_box_domain_from_center_and_size():
    """Test BoxDomain creation from center and size."""
    center = Point3D(1.0, 2.0, 3.0)
    box = BoxDomain.from_center_and_size(
        center=center,
        width=0.1,
        height=0.2,
        depth=0.3,
    )
    
    assert box.x_min == pytest.approx(0.95)
    assert box.x_max == pytest.approx(1.05)
    assert box.y_min == pytest.approx(1.9)
    assert box.y_max == pytest.approx(2.1)
    assert box.z_min == pytest.approx(2.85)
    assert box.z_max == pytest.approx(3.15)


def test_box_domain_invalid_dimensions():
    """Test BoxDomain validation."""
    with pytest.raises(ValueError, match="x_min.*must be less than x_max"):
        BoxDomain(
            x_min=0.05, x_max=-0.05,
            y_min=-0.03, y_max=0.03,
            z_min=-0.02, z_max=0.02,
        )
    
    with pytest.raises(ValueError, match="y_min.*must be less than y_max"):
        BoxDomain(
            x_min=-0.05, x_max=0.05,
            y_min=0.03, y_max=-0.03,
            z_min=-0.02, z_max=0.02,
        )
    
    with pytest.raises(ValueError, match="z_min.*must be less than z_max"):
        BoxDomain(
            x_min=-0.05, x_max=0.05,
            y_min=-0.03, y_max=0.03,
            z_min=0.02, z_max=-0.02,
        )


def test_box_domain_contains():
    """Test BoxDomain.contains()."""
    box = BoxDomain(
        x_min=-0.05, x_max=0.05,
        y_min=-0.03, y_max=0.03,
        z_min=-0.02, z_max=0.02,
    )
    
    assert box.contains(Point3D(0.0, 0.0, 0.0))
    assert box.contains(Point3D(0.04, 0.02, 0.01))
    assert box.contains(Point3D(-0.04, -0.02, -0.01))
    
    assert box.contains(Point3D(0.05, 0.0, 0.0))
    assert box.contains(Point3D(0.0, 0.03, 0.0))
    assert box.contains(Point3D(0.0, 0.0, 0.02))
    
    assert not box.contains(Point3D(0.06, 0.0, 0.0))
    assert not box.contains(Point3D(0.0, 0.04, 0.0))
    assert not box.contains(Point3D(0.0, 0.0, 0.03))
    assert not box.contains(Point3D(-0.06, 0.0, 0.0))


def test_box_domain_project_inside():
    """Test BoxDomain.project_inside()."""
    box = BoxDomain(
        x_min=-0.05, x_max=0.05,
        y_min=-0.03, y_max=0.03,
        z_min=-0.02, z_max=0.02,
    )
    
    p_inside = Point3D(0.01, 0.01, 0.01)
    projected = box.project_inside(p_inside)
    assert projected.x == pytest.approx(0.01)
    assert projected.y == pytest.approx(0.01)
    assert projected.z == pytest.approx(0.01)
    
    p_outside_x = Point3D(0.1, 0.0, 0.0)
    projected = box.project_inside(p_outside_x)
    assert projected.x < 0.05  # Clamped with margin
    assert projected.y == pytest.approx(0.0)
    assert projected.z == pytest.approx(0.0)
    
    p_outside = Point3D(0.1, 0.05, 0.03)
    projected = box.project_inside(p_outside)
    assert projected.x < 0.05
    assert projected.y < 0.03
    assert projected.z < 0.02


def test_box_domain_distance_to_boundary():
    """Test BoxDomain.distance_to_boundary()."""
    box = BoxDomain(
        x_min=-0.05, x_max=0.05,
        y_min=-0.03, y_max=0.03,
        z_min=-0.02, z_max=0.02,
    )
    
    center = Point3D(0.0, 0.0, 0.0)
    dist = box.distance_to_boundary(center)
    assert dist == pytest.approx(0.02)  # Closest face is z
    
    p_near_x = Point3D(0.04, 0.0, 0.0)
    dist = box.distance_to_boundary(p_near_x)
    assert dist == pytest.approx(0.01)  # Distance to x_max
    
    p_near_y = Point3D(0.0, 0.025, 0.0)
    dist = box.distance_to_boundary(p_near_y)
    assert dist == pytest.approx(0.005)  # Distance to y_max


def test_box_domain_sample_points():
    """Test BoxDomain.sample_points()."""
    box = BoxDomain(
        x_min=-0.05, x_max=0.05,
        y_min=-0.03, y_max=0.03,
        z_min=-0.02, z_max=0.02,
    )
    
    points = box.sample_points(100, seed=42)
    
    assert points.shape == (100, 3)
    
    for point_arr in points:
        point = Point3D.from_array(point_arr)
        assert box.contains(point)
    
    assert np.all(points[:, 0] >= -0.05)
    assert np.all(points[:, 0] <= 0.05)
    assert np.all(points[:, 1] >= -0.03)
    assert np.all(points[:, 1] <= 0.03)
    assert np.all(points[:, 2] >= -0.02)
    assert np.all(points[:, 2] <= 0.02)
    
    points2 = box.sample_points(100, seed=42)
    np.testing.assert_array_equal(points, points2)


def test_box_domain_get_bounds():
    """Test BoxDomain.get_bounds()."""
    box = BoxDomain(
        x_min=-0.05, x_max=0.05,
        y_min=-0.03, y_max=0.03,
        z_min=-0.02, z_max=0.02,
    )
    
    bounds = box.get_bounds()
    assert bounds == (-0.05, 0.05, -0.03, 0.03, -0.02, 0.02)


def test_box_domain_serialization():
    """Test BoxDomain to_dict() and from_dict()."""
    box = BoxDomain(
        x_min=-0.05, x_max=0.05,
        y_min=-0.03, y_max=0.03,
        z_min=-0.02, z_max=0.02,
    )
    
    d = box.to_dict()
    assert d["type"] == "box"
    assert d["x_min"] == -0.05
    assert d["x_max"] == 0.05
    assert d["y_min"] == -0.03
    assert d["y_max"] == 0.03
    assert d["z_min"] == -0.02
    assert d["z_max"] == 0.02
    
    box2 = BoxDomain.from_dict(d)
    assert box2.x_min == box.x_min
    assert box2.x_max == box.x_max
    assert box2.y_min == box.y_min
    assert box2.y_max == box.y_max
    assert box2.z_min == box.z_min
    assert box2.z_max == box.z_max


def test_box_domain_with_network():
    """Test BoxDomain with VascularNetwork."""
    from vascular_lib.ops import create_network, add_inlet, add_outlet
    
    box = BoxDomain(
        x_min=-0.05, x_max=0.05,
        y_min=-0.03, y_max=0.03,
        z_min=-0.02, z_max=0.02,
    )
    
    network = create_network(box)
    
    assert network.domain == box
    
    add_inlet(
        network,
        position=(-0.04, 0.0, 0.0),
        direction=(1, 0, 0),
        radius=0.003,
    )
    
    add_outlet(
        network,
        position=(0.04, 0.0, 0.0),
        direction=(-1, 0, 0),
        radius=0.003,
    )
    
    assert len(network.nodes) == 2
    assert len(network.segments) == 0
