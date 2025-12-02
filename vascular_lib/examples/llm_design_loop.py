"""
Example: LLM-driven iterative vascular network design.

This demonstrates the core design loop that an LLM would follow:
1. Create network with domain
2. Add inlets and outlets
3. Query current state
4. Take design actions (grow, bifurcate, space colonization)
5. Get structured feedback
6. Iterate based on feedback
"""

import numpy as np
from vascular_lib import (
    create_network,
    add_inlet,
    add_outlet,
    grow_branch,
    bifurcate,
    space_colonization_step,
    get_leaf_nodes,
    compute_coverage,
    estimate_flows,
    check_hemodynamic_plausibility,
    save_json,
    load_json,
)
from vascular_lib.core import EllipsoidDomain
from vascular_lib.ops.space_colonization import SpaceColonizationParams


def example_basic_construction():
    """Example 1: Basic network construction."""
    print("=" * 60)
    print("Example 1: Basic Network Construction")
    print("=" * 60)
    
    domain = EllipsoidDomain(
        semi_axis_a=0.12,  # 12 cm
        semi_axis_b=0.10,  # 10 cm
        semi_axis_c=0.08,  # 8 cm
    )
    
    network = create_network(domain, seed=42)
    print(f"✓ Created network with domain bounds: {domain.get_bounds()}")
    
    result = add_inlet(
        network,
        position=(-0.10, 0.0, 0.0),
        direction=(1.0, 0.0, 0.0),
        radius=0.005,  # 5mm
        vessel_type="arterial",
    )
    
    if result.is_success():
        inlet_id = result.new_ids['node']
        print(f"✓ Added arterial inlet (node {inlet_id}): {result.message}")
    else:
        print(f"✗ Failed to add inlet: {result.message}")
        return
    
    result = add_outlet(
        network,
        position=(0.10, 0.0, 0.0),
        direction=(-1.0, 0.0, 0.0),
        radius=0.006,  # 6mm
        vessel_type="venous",
    )
    
    if result.is_success():
        outlet_id = result.new_ids['node']
        print(f"✓ Added venous outlet (node {outlet_id}): {result.message}")
    else:
        print(f"✗ Failed to add outlet: {result.message}")
        return
    
    print(f"\nNetwork state: {len(network.nodes)} nodes, {len(network.segments)} segments")
    return network, inlet_id, outlet_id


def example_manual_growth(network, inlet_id):
    """Example 2: Manual branch growth and bifurcation."""
    print("\n" + "=" * 60)
    print("Example 2: Manual Growth and Bifurcation")
    print("=" * 60)
    
    result = grow_branch(
        network,
        from_node_id=inlet_id,
        length=0.02,  # 2cm
        direction=(1.0, 0.1, 0.0),
        target_radius=0.004,
    )
    
    if result.is_success():
        tip_id = result.new_ids['node']
        print(f"✓ Grew branch: {result.message}")
        if result.warnings:
            print(f"  Warnings: {result.warnings}")
    else:
        print(f"✗ Growth failed: {result.message}")
        return
    
    result = bifurcate(
        network,
        at_node_id=tip_id,
        child_lengths=(0.015, 0.015),  # 1.5cm each
        angle_deg=45.0,
    )
    
    if result.is_success():
        child_ids = result.new_ids['nodes']
        print(f"✓ Created bifurcation: {result.message}")
        print(f"  Child nodes: {child_ids}")
    else:
        print(f"✗ Bifurcation failed: {result.message}")
    
    print(f"\nNetwork state: {len(network.nodes)} nodes, {len(network.segments)} segments")


def example_space_colonization(network, inlet_id):
    """Example 3: Space colonization for organic growth."""
    print("\n" + "=" * 60)
    print("Example 3: Space Colonization Growth")
    print("=" * 60)
    
    tissue_points = network.domain.sample_points(n_points=500, seed=42)
    print(f"✓ Sampled {len(tissue_points)} tissue points")
    
    params = SpaceColonizationParams(
        influence_radius=0.020,  # 2cm
        kill_radius=0.005,  # 5mm
        step_size=0.008,  # 8mm
        min_radius=0.0005,  # 0.5mm
        vessel_type="arterial",
        max_steps=50,
    )
    
    result = space_colonization_step(
        network,
        tissue_points=tissue_points,
        params=params,
        seed=42,
    )
    
    if result.is_success():
        print(f"✓ Space colonization: {result.message}")
        print(f"  Metadata: {result.metadata}")
    else:
        print(f"✗ Space colonization failed: {result.message}")
    
    print(f"\nNetwork state: {len(network.nodes)} nodes, {len(network.segments)} segments")


def example_coverage_analysis(network):
    """Example 4: Coverage analysis."""
    print("\n" + "=" * 60)
    print("Example 4: Coverage Analysis")
    print("=" * 60)
    
    test_points = network.domain.sample_points(n_points=1000, seed=123)
    print(f"✓ Sampled {len(test_points)} test points")
    
    coverage = compute_coverage(
        network,
        tissue_points=test_points,
        diffusion_distance=0.005,  # 5mm
        vessel_type="arterial",
    )
    
    print(f"\nCoverage Results:")
    print(f"  Fraction covered: {coverage['fraction_covered']:.1%}")
    print(f"  Covered points: {coverage['covered_count']}/{coverage['total_points']}")
    print(f"  Mean distance to vessel: {coverage['mean_coverage_distance']*1000:.2f} mm")
    print(f"  Max distance to vessel: {coverage['max_coverage_distance']*1000:.2f} mm")
    
    if coverage['uncovered_regions']:
        print(f"\n  Uncovered regions: {len(coverage['uncovered_regions'])}")
        for i, region in enumerate(coverage['uncovered_regions'][:3]):
            print(f"    Region {i+1}: {region['point_count']} points near node {region['nearest_node']}")


def example_flow_analysis(network, inlet_id, outlet_id):
    """Example 5: Flow analysis."""
    print("\n" + "=" * 60)
    print("Example 5: Flow Analysis")
    print("=" * 60)
    
    inlet_pressures = {inlet_id: 13000.0}  # ~100 mmHg
    outlet_pressures = {outlet_id: 2000.0}  # ~15 mmHg
    
    flow_result = estimate_flows(
        network,
        inlet_pressures=inlet_pressures,
        outlet_pressures=outlet_pressures,
    )
    
    print(f"\nFlow Results:")
    print(f"  Total inlet flow: {flow_result['total_inlet_flow']*1e6:.2f} mL/s")
    print(f"  Total outlet flow: {flow_result['total_outlet_flow']*1e6:.2f} mL/s")
    print(f"  Flow balance error: {flow_result['flow_balance_error']:.1%}")
    
    warnings = check_hemodynamic_plausibility(network, flow_result)
    
    if warnings:
        print(f"\n  Hemodynamic warnings ({len(warnings)}):")
        for warning in warnings[:5]:
            print(f"    - {warning}")
    else:
        print(f"\n  ✓ No hemodynamic issues detected")


def example_serialization(network):
    """Example 6: Save and load network."""
    print("\n" + "=" * 60)
    print("Example 6: Serialization")
    print("=" * 60)
    
    output_path = "/tmp/example_network.json"
    save_json(network, output_path)
    print(f"✓ Saved network to {output_path}")
    
    loaded_network = load_json(output_path)
    print(f"✓ Loaded network from {output_path}")
    print(f"  Nodes: {len(loaded_network.nodes)}")
    print(f"  Segments: {len(loaded_network.segments)}")
    
    assert len(loaded_network.nodes) == len(network.nodes)
    assert len(loaded_network.segments) == len(network.segments)
    print(f"✓ Verification passed")


def example_query_operations(network, inlet_id):
    """Example 7: Query operations."""
    print("\n" + "=" * 60)
    print("Example 7: Query Operations")
    print("=" * 60)
    
    from vascular_lib import get_leaf_nodes, get_paths_from_inlet, measure_segment_lengths
    
    arterial_leaves = get_leaf_nodes(network, vessel_type="arterial")
    print(f"✓ Found {len(arterial_leaves)} arterial leaf nodes")
    
    paths = get_paths_from_inlet(network, inlet_id)
    print(f"✓ Found {len(paths)} paths from inlet")
    if paths:
        print(f"  Example path length: {len(paths[0])} nodes")
    
    stats = measure_segment_lengths(network, vessel_type="arterial")
    print(f"\nSegment Length Statistics:")
    print(f"  Count: {stats['count']}")
    print(f"  Mean: {stats['mean']*1000:.2f} mm")
    print(f"  Std: {stats['std']*1000:.2f} mm")
    print(f"  Range: [{stats['min']*1000:.2f}, {stats['max']*1000:.2f}] mm")
    print(f"  Total: {stats['total']*100:.2f} cm")


def main():
    """Run all examples."""
    print("\n" + "=" * 60)
    print("LLM-Driven Vascular Network Design Examples")
    print("=" * 60)
    
    result = example_basic_construction()
    if result is None:
        return
    network, inlet_id, outlet_id = result
    
    example_manual_growth(network, inlet_id)
    
    example_space_colonization(network, inlet_id)
    
    example_coverage_analysis(network)
    
    example_flow_analysis(network, inlet_id, outlet_id)
    
    example_serialization(network)
    
    example_query_operations(network, inlet_id)
    
    print("\n" + "=" * 60)
    print("All examples completed successfully!")
    print("=" * 60)


if __name__ == "__main__":
    main()
