"""
Advanced example using individual functions from the vascular network package.

This example demonstrates:
1. Using individual mesh processing functions
2. Running analysis functions separately
3. Custom visualization
"""

from vascular_network.io import load_stl_mesh
from vascular_network.mesh import (
    basic_clean,
    voxel_remesh_and_smooth,
    meshfix_repair,
    compute_diagnostics,
    compute_surface_quality,
)
from vascular_network.analysis import (
    analyze_connectivity_voxel,
    extract_centerline_graph,
    compute_poiseuille_network,
)
from vascular_network.visualization import (
    plot_surface_quality,
    plot_centerline_graph_2d,
    plot_centerline_scalar,
    summarize_poiseuille,
)

print("Loading mesh...")
mesh = load_stl_mesh("input.stl")

diag_before = compute_diagnostics(mesh)
print(f"Before: Watertight={diag_before.watertight}, Volume={diag_before.volume}")

print("Cleaning mesh...")
mesh_clean = basic_clean(mesh)

print("Voxel remeshing...")
mesh_voxel = voxel_remesh_and_smooth(
    mesh_clean,
    pitch=0.1,
    smooth_iters=40,
)

print("Repairing mesh...")
mesh_repaired = meshfix_repair(mesh_voxel)

diag_after = compute_diagnostics(mesh_repaired)
surf_after = compute_surface_quality(mesh_repaired)
print(f"After: Watertight={diag_after.watertight}, Volume={diag_after.volume}")

print("Analyzing connectivity...")
connectivity_info, fluid_mask, origin, pitch = analyze_connectivity_voxel(
    mesh_repaired,
    pitch=0.1,
)
print(f"Fluid components: {connectivity_info['num_fluid_components']}")
print(f"Reachable fraction: {connectivity_info['reachable_fraction']:.2%}")

print("Extracting centerline...")
centerline_graph, meta = extract_centerline_graph(fluid_mask, origin, pitch)
print(f"Centerline nodes: {meta['num_nodes']}, edges: {meta['num_edges']}")

print("Running Poiseuille flow analysis...")
cfd_results = compute_poiseuille_network(
    centerline_graph,
    mu=1.0,
    inlet_nodes=None,
    outlet_nodes=None,
    pin=1.0,
    pout=0.0,
)

summarize_poiseuille(cfd_results["G"])

print("Generating visualizations...")
plot_surface_quality(mesh_repaired, "Repaired Mesh")
plot_centerline_graph_2d(centerline_graph, plane="xz")
plot_centerline_scalar(cfd_results["G"], edge_attr="Q", log_abs=True)

print("\nProcessing complete!")
