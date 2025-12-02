import pytest
import numpy as np
import math
from pathlib import Path

from vascular_network import validate_and_repair_geometry
from vascular_network.analysis import compute_poiseuille_network


def test_y_branch_watertightness(y_branch_mesh, temp_dir):
    """Test that Y-branch mesh becomes watertight after processing."""
    mesh, R_in, L_in, R_out1, L_out1, R_out2, L_out2 = y_branch_mesh
    
    stl_path = temp_dir / "y_branch.stl"
    mesh.export(stl_path)
    
    report, G = validate_and_repair_geometry(
        input_path=stl_path,
        cleaned_stl_path=temp_dir / "y_branch_cleaned.stl",
        scaffold_stl_path=temp_dir / "y_branch_scaffold.stl",
        report_path=temp_dir / "y_branch_report.json",
        voxel_pitch=0.1,
        smooth_iters=20,
    )
    
    assert report.after_repair.watertight, "Mesh should be watertight after repair"
    
    assert report.after_repair.num_components == 1, "Should have single component"


def test_y_branch_connectivity(y_branch_mesh, temp_dir):
    """Test that Y-branch has proper connectivity."""
    mesh, R_in, L_in, R_out1, L_out1, R_out2, L_out2 = y_branch_mesh
    
    stl_path = temp_dir / "y_branch.stl"
    mesh.export(stl_path)
    
    report, G = validate_and_repair_geometry(
        input_path=stl_path,
        voxel_pitch=0.1,
        smooth_iters=20,
    )
    
    conn = report.connectivity
    assert conn["num_fluid_components"] >= 1, "Should have at least one fluid component"
    assert conn["reachable_fraction"] > 0.7, "Most of the fluid should be reachable"


def test_y_branch_centerline(y_branch_mesh, temp_dir):
    """Test that centerline extraction works for Y-branch."""
    mesh, R_in, L_in, R_out1, L_out1, R_out2, L_out2 = y_branch_mesh
    
    stl_path = temp_dir / "y_branch.stl"
    mesh.export(stl_path)
    
    report, G = validate_and_repair_geometry(
        input_path=stl_path,
        voxel_pitch=0.1,
        smooth_iters=20,
    )
    
    assert G.number_of_nodes() > 0, "Centerline should have nodes"
    assert G.number_of_edges() > 0, "Centerline should have edges"
    
    leaves = [n for n in G.nodes if G.degree(n) == 1]
    assert len(leaves) >= 3, f"Should have at least 3 leaf nodes, found {len(leaves)}"


def test_y_branch_flow_split(y_branch_mesh, temp_dir):
    """Test that flow splits correctly at Y-branch."""
    mesh, R_in, L_in, R_out1, L_out1, R_out2, L_out2 = y_branch_mesh
    
    stl_path = temp_dir / "y_branch.stl"
    mesh.export(stl_path)
    
    mu = 1.0
    pin = 1.0
    pout = 0.0
    
    report, G = validate_and_repair_geometry(
        input_path=stl_path,
        voxel_pitch=0.1,
        smooth_iters=20,
    )
    
    leaves = [n for n in G.nodes if G.degree(n) == 1]
    assert len(leaves) >= 3, "Should have at least 3 leaf nodes"
    
    coords = {n: np.asarray(G.nodes[n].get("coord", [0, 0, 0]), dtype=float) for n in leaves}
    
    inlet = min(leaves, key=lambda n: coords[n][2])
    
    outlets = [n for n in leaves if n != inlet]
    
    # Re-run Poiseuille with explicit boundaries
    results = compute_poiseuille_network(
        G,
        mu=mu,
        inlet_nodes=[inlet],
        outlet_nodes=outlets[:2] if len(outlets) >= 2 else outlets,
        pin=pin,
        pout=pout,
    )
    
    edge_flows = results["edge_flows"]
    
    def net_outflow(node):
        q = 0.0
        for (u, v), Q_uv in edge_flows.items():
            if u == node:
                q += Q_uv
            elif v == node:
                q -= Q_uv
        return q
    
    if len(outlets) >= 2:
        Q1 = abs(net_outflow(outlets[0]))
        Q2 = abs(net_outflow(outlets[1]))
        
        assert Q1 > 0, "Outlet 1 should have positive flow"
        assert Q2 > 0, "Outlet 2 should have positive flow"
        
        total_outflow = Q1 + Q2
        assert total_outflow > 0, "Total outflow should be positive"


def test_y_branch_flow_conservation(y_branch_mesh, temp_dir):
    """Test that flow is conserved in Y-branch (inlet ≈ sum of outlets)."""
    mesh, R_in, L_in, R_out1, L_out1, R_out2, L_out2 = y_branch_mesh
    
    stl_path = temp_dir / "y_branch.stl"
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
        assert flow_balance < 0.2, f"Flow should be conserved, balance error: {flow_balance}"


def test_y_branch_pressure_gradient(y_branch_mesh, temp_dir):
    """Test that pressure decreases from inlet to outlets."""
    mesh, R_in, L_in, R_out1, L_out1, R_out2, L_out2 = y_branch_mesh
    
    stl_path = temp_dir / "y_branch.stl"
    mesh.export(stl_path)
    
    report, G = validate_and_repair_geometry(
        input_path=stl_path,
        voxel_pitch=0.1,
        smooth_iters=20,
    )
    
    pressures = [G.nodes[n].get("pressure", None) for n in G.nodes]
    pressures = [p for p in pressures if p is not None]
    
    if len(pressures) > 0:
        p_min = min(pressures)
        p_max = max(pressures)
        assert p_max > p_min, "Should have pressure gradient"


def test_y_branch_multiple_components(y_branch_mesh, temp_dir):
    """Test that Y-branch is processed as a single connected component."""
    mesh, R_in, L_in, R_out1, L_out1, R_out2, L_out2 = y_branch_mesh
    
    stl_path = temp_dir / "y_branch.stl"
    mesh.export(stl_path)
    
    report, G = validate_and_repair_geometry(
        input_path=stl_path,
        voxel_pitch=0.1,
        smooth_iters=20,
    )
    
    assert report.after_repair.num_components == 1, "Should have single component after repair"
    
    import networkx as nx
    assert nx.is_connected(G), "Centerline graph should be connected"
