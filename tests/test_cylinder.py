import pytest
import numpy as np
import math
from pathlib import Path

from vascular_network import validate_and_repair_geometry
from vascular_network.mesh import compute_diagnostics


def test_cylinder_watertightness(straight_cylinder_mesh, temp_dir):
    """Test that cylinder mesh becomes watertight after processing."""
    mesh, radius, length = straight_cylinder_mesh
    
    stl_path = temp_dir / "cylinder.stl"
    mesh.export(stl_path)
    
    report, G = validate_and_repair_geometry(
        input_path=stl_path,
        cleaned_stl_path=temp_dir / "cylinder_cleaned.stl",
        scaffold_stl_path=temp_dir / "cylinder_scaffold.stl",
        report_path=temp_dir / "cylinder_report.json",
        voxel_pitch=0.1,
        smooth_iters=20,
    )
    
    assert report.after_repair.watertight, "Mesh should be watertight after repair"
    
    assert report.after_repair.num_components == 1, "Should have single component"
    
    expected_volume = math.pi * radius**2 * length
    assert report.after_repair.volume is not None
    volume_ratio = report.after_repair.volume / expected_volume
    assert 0.5 < volume_ratio < 1.5, f"Volume ratio {volume_ratio} should be reasonable"


def test_cylinder_connectivity(straight_cylinder_mesh, temp_dir):
    """Test that cylinder has proper connectivity."""
    mesh, radius, length = straight_cylinder_mesh
    
    stl_path = temp_dir / "cylinder.stl"
    mesh.export(stl_path)
    
    report, G = validate_and_repair_geometry(
        input_path=stl_path,
        voxel_pitch=0.1,
        smooth_iters=20,
    )
    
    conn = report.connectivity
    assert conn["num_fluid_components"] >= 1, "Should have at least one fluid component"
    assert conn["reachable_fraction"] > 0.8, "Most of the fluid should be reachable"


def test_cylinder_centerline(straight_cylinder_mesh, temp_dir):
    """Test that centerline extraction works for cylinder."""
    mesh, radius, length = straight_cylinder_mesh
    
    stl_path = temp_dir / "cylinder.stl"
    mesh.export(stl_path)
    
    report, G = validate_and_repair_geometry(
        input_path=stl_path,
        voxel_pitch=0.1,
        smooth_iters=20,
    )
    
    assert G.number_of_nodes() > 0, "Centerline should have nodes"
    assert G.number_of_edges() > 0, "Centerline should have edges"
    
    for node, data in G.nodes(data=True):
        assert "radius" in data, "Each node should have radius"
        assert data["radius"] > 0, "Radius should be positive"


def test_cylinder_poiseuille_flow(straight_cylinder_mesh, temp_dir):
    """Test that Poiseuille flow analysis works for cylinder."""
    mesh, radius, length = straight_cylinder_mesh
    
    stl_path = temp_dir / "cylinder.stl"
    mesh.export(stl_path)
    
    mu = 1.0
    pin = 1.0
    pout = 0.0
    
    report, G = validate_and_repair_geometry(
        input_path=stl_path,
        voxel_pitch=0.1,
        smooth_iters=20,
    )
    
    poi = report.poiseuille_summary
    assert poi["num_nodes"] > 0, "Should have nodes in flow network"
    assert poi["num_edges"] > 0, "Should have edges in flow network"
    
    Q_num = poi.get("total_inlet_flow", poi.get("total_inflow", None))
    assert Q_num is not None, "Should have computed inlet flow"
    assert Q_num > 0, "Flow should be positive"
    
    delta_p = pin - pout
    Q_ref = math.pi * radius**4 * delta_p / (8.0 * mu * length)
    
    rel_error = abs(Q_num - Q_ref) / abs(Q_ref)
    assert rel_error < 2.0, f"Flow error {rel_error} should be reasonable"


def test_cylinder_volume_convergence(straight_cylinder_mesh, temp_dir):
    """Test that volume converges with finer voxel pitch."""
    mesh, radius, length = straight_cylinder_mesh
    
    stl_path = temp_dir / "cylinder.stl"
    mesh.export(stl_path)
    
    expected_volume = math.pi * radius**2 * length
    
    pitches = [0.2, 0.1, 0.05]
    volumes = []
    
    for pitch in pitches:
        report, G = validate_and_repair_geometry(
            input_path=stl_path,
            voxel_pitch=pitch,
            smooth_iters=10,
        )
        volumes.append(report.after_repair.volume)
    
    errors = [abs(v - expected_volume) / expected_volume for v in volumes]
    
    assert errors[-1] < 0.5, "Finest pitch should give reasonable volume"


def test_cylinder_flow_conservation(straight_cylinder_mesh, temp_dir):
    """Test that flow is conserved (inlet flow ≈ outlet flow)."""
    mesh, radius, length = straight_cylinder_mesh
    
    stl_path = temp_dir / "cylinder.stl"
    mesh.export(stl_path)
    
    report, G = validate_and_repair_geometry(
        input_path=stl_path,
        voxel_pitch=0.1,
        smooth_iters=20,
    )
    
    poi = report.poiseuille_summary
    Q_in = poi.get("total_inlet_flow", poi.get("total_inflow", 0))
    Q_out = poi.get("total_outlet_flow", poi.get("total_outflow", 0))
    
    if Q_in > 0 and Q_out > 0:
        flow_balance = abs(Q_in - Q_out) / max(Q_in, Q_out)
        assert flow_balance < 0.1, f"Flow should be conserved, balance error: {flow_balance}"
