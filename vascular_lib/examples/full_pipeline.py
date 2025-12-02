"""
Complete pipeline example: Design → Export STL → Repair → Flow → Report

This demonstrates the full workflow for LLM-driven vascular design.
"""

import numpy as np
from vascular_lib.core.domain import EllipsoidDomain
from vascular_lib.ops.build import create_network, add_inlet, add_outlet
from vascular_lib.ops.growth import grow_branch, bifurcate
from vascular_lib.ops.space_colonization import space_colonization_step, SpaceColonizationParams
from vascular_lib.ops.collision import get_collisions, avoid_collisions
from vascular_lib.analysis.solver import solve_flow, check_flow_plausibility
from vascular_lib.adapters.mesh_adapter import export_stl
from vascular_lib.adapters.report_adapter import make_full_report
from vascular_lib.io.serialize import save_json
from vascular_lib.core.types import Direction3D


def main():
    """Run complete pipeline."""
    print("=" * 60)
    print("VASCULAR NETWORK DESIGN PIPELINE")
    print("=" * 60)
    
    print("\n[1/8] Creating liver domain and network...")
    domain = EllipsoidDomain(
        center=(0, 0, 0),
        semi_axes=(0.12, 0.1, 0.08),  # 12cm x 10cm x 8cm liver
    )
    
    network = create_network(
        domain,
        metadata={
            "organ": "liver",
            "species": "human",
            "units": "meters",
        },
        seed=42,
    )
    
    print(f"   Domain: {domain.semi_axes} m")
    
    print("\n[2/8] Adding inlet and outlet nodes...")
    inlet_result = add_inlet(
        network,
        position=(-0.10, 0, 0),
        direction=Direction3D(x=1, y=0, z=0),
        radius=0.005,  # 5mm arterial inlet
        vessel_type="arterial",
    )
    
    outlet_result = add_outlet(
        network,
        position=(0.10, 0, 0),
        direction=Direction3D(x=-1, y=0, z=0),
        radius=0.006,  # 6mm venous outlet
        vessel_type="venous",
    )
    
    print(f"   Inlet node: {inlet_result.new_ids['node_id']}")
    print(f"   Outlet node: {outlet_result.new_ids['node_id']}")
    
    print("\n[3/8] Growing arterial tree with space colonization...")
    
    np.random.seed(123)
    tissue_points = []
    for _ in range(200):
        x = np.random.uniform(-0.08, 0.02)
        y = np.random.uniform(-0.08, 0.08)
        z = np.random.uniform(-0.06, 0.06)
        if domain.contains_point((x, y, z)):
            tissue_points.append([x, y, z])
    tissue_points = np.array(tissue_points)
    
    arterial_params = SpaceColonizationParams(
        vessel_type="arterial",
        influence_radius=0.025,
        kill_radius=0.008,
        step_size=0.01,
        max_steps=15,
        min_radius=0.0003,
        taper_factor=0.95,
    )
    
    arterial_result = space_colonization_step(
        network,
        tissue_points,
        arterial_params,
        seed=42,
    )
    
    print(f"   Arterial tree: {arterial_result.metadata['nodes_created']} nodes, "
          f"{arterial_result.metadata['segments_created']} segments")
    print(f"   Coverage: {arterial_result.metadata['coverage_fraction']:.1%}")
    
    print("\n[4/8] Growing venous tree...")
    
    venous_tissue = []
    for _ in range(200):
        x = np.random.uniform(-0.02, 0.08)
        y = np.random.uniform(-0.08, 0.08)
        z = np.random.uniform(-0.06, 0.06)
        if domain.contains_point((x, y, z)):
            venous_tissue.append([x, y, z])
    venous_tissue = np.array(venous_tissue)
    
    venous_params = SpaceColonizationParams(
        vessel_type="venous",
        influence_radius=0.025,
        kill_radius=0.008,
        step_size=0.01,
        max_steps=15,
        min_radius=0.0003,
        taper_factor=0.95,
    )
    
    venous_result = space_colonization_step(
        network,
        venous_tissue,
        venous_params,
        seed=43,
    )
    
    print(f"   Venous tree: {venous_result.metadata['nodes_created']} nodes, "
          f"{venous_result.metadata['segments_created']} segments")
    
    print("\n[5/8] Checking for collisions...")
    collision_result = get_collisions(network, min_clearance=0.001)
    
    if collision_result.metadata['count'] > 0:
        print(f"   Found {collision_result.metadata['count']} collisions")
        print("   Attempting repair by shrinking...")
        
        repair_result = avoid_collisions(
            network,
            min_clearance=0.001,
            repair_strategy="shrink",
        )
        
        if repair_result.is_success():
            print(f"   ✓ Repair successful")
        else:
            print(f"   ⚠ Partial repair: {len(repair_result.warnings)} warnings")
    else:
        print("   ✓ No collisions detected")
    
    print("\n[6/8] Solving hemodynamics...")
    flow_result = solve_flow(
        network,
        pin=13000.0,  # ~100 mmHg
        pout=2000.0,   # ~15 mmHg
        mu=1.0e-3,     # Blood viscosity
    )
    
    if flow_result.is_success():
        print(f"   ✓ Flow solved")
        print(f"   Inlet flow: {flow_result.metadata['inlet_flow']*1e6:.2f} mL/s")
        print(f"   Outlet flow: {flow_result.metadata['outlet_flow']*1e6:.2f} mL/s")
        print(f"   Balance error: {flow_result.metadata['flow_balance_error']:.2%}")
    else:
        print(f"   ✗ Flow solver failed: {flow_result.message}")
    
    plausibility_result = check_flow_plausibility(network)
    print(f"   Max Reynolds: {plausibility_result.metadata['max_reynolds']:.0f}")
    print(f"   Laminar: {plausibility_result.metadata['is_laminar']}")
    
    print("\n[7/8] Exporting STL mesh...")
    stl_result = export_stl(
        network,
        output_path="/tmp/vascular_network.stl",
        mode="fast",
        repair=True,
        radial_resolution=8,
    )
    
    if stl_result.is_success():
        print(f"   ✓ Exported to {stl_result.metadata['output_path']}")
        print(f"   Watertight: {stl_result.metadata.get('is_watertight', False)}")
        if stl_result.metadata.get('was_repaired'):
            print(f"   (mesh was repaired)")
    else:
        print(f"   ✗ Export failed: {stl_result.message}")
    
    print("\n[8/8] Generating comprehensive report...")
    report = make_full_report(
        network,
        include_flow=True,
        include_geometry=True,
        include_coverage=False,
    )
    
    save_json(report, "/tmp/vascular_report.json")
    print(f"   ✓ Report saved to /tmp/vascular_report.json")
    
    print("\n" + "=" * 60)
    print("PIPELINE COMPLETE")
    print("=" * 60)
    print(f"\nNetwork Summary:")
    print(f"  Nodes: {len(network.nodes)}")
    print(f"  Segments: {len(network.segments)}")
    print(f"  Arterial segments: {sum(1 for s in network.segments.values() if s.vessel_type == 'arterial')}")
    print(f"  Venous segments: {sum(1 for s in network.segments.values() if s.vessel_type == 'venous')}")
    
    if 'quality_flags' in report:
        print(f"\nQuality Flags:")
        for key, value in report['quality_flags'].items():
            print(f"  {key}: {value}")
    
    print(f"\nSummary: {report.get('summary', 'N/A')}")
    
    print("\n" + "=" * 60)
    print("Files generated:")
    print("  - /tmp/vascular_network.stl (3D mesh)")
    print("  - /tmp/vascular_report.json (comprehensive report)")
    print("=" * 60)


if __name__ == "__main__":
    main()
