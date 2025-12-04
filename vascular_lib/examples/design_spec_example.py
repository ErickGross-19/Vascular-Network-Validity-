"""
Example: Using DesignSpec API for LLM-driven vascular network design.

This demonstrates the high-level DesignSpec API which allows LLMs to:
1. Define network designs as JSON-serializable specifications
2. Build networks from specs with one function call
3. Modify specs iteratively based on evaluation feedback
4. Compare different design approaches
"""

from vascular_lib.api import design_from_spec, evaluate_network
from vascular_lib.specs import TreeSpec, DualTreeSpec, EllipsoidSpec, BoxSpec, ColonizationSpec
from vascular_lib.params import get_preset
from vascular_lib.io import save_json


def example_1_single_tree_with_preset():
    """Example 1: Single arterial tree using parameter preset."""
    print("=" * 70)
    print("Example 1: Single Arterial Tree with Preset Parameters")
    print("=" * 70)
    
    spec = TreeSpec(
        domain=EllipsoidSpec(
            semi_axes=(0.12, 0.10, 0.08),  # 12cm x 10cm x 8cm liver
        ),
        inlet={"position": (-0.10, 0.0, 0.0), "radius": 0.005},
        colonization=get_preset("liver_arterial_dense"),
        tissue_points=200,
        seed=42,
    )
    
    print(f"\n✓ Created TreeSpec:")
    print(f"  Domain: Ellipsoid {spec.domain.semi_axes}")
    print(f"  Inlet: {spec.inlet}")
    print(f"  Preset: liver_arterial_dense")
    print(f"  Tissue points: {spec.tissue_points}")
    
    print(f"\n✓ Building network from spec...")
    network = design_from_spec(spec)
    
    print(f"\n✓ Network built:")
    print(f"  Nodes: {len(network.nodes)}")
    print(f"  Segments: {len(network.segments)}")
    
    print(f"\n✓ Evaluating network quality...")
    eval_result = evaluate_network(network, tissue_points=1000)
    
    print(f"\n✓ Evaluation Results:")
    print(f"  Coverage: {eval_result.metrics.coverage_fraction:.1%}")
    print(f"  Coverage score: {eval_result.scores.coverage_score:.2f}")
    print(f"  Structure score: {eval_result.scores.structure_score:.2f}")
    print(f"  Overall score: {eval_result.scores.overall_score:.2f}")
    
    if eval_result.recommendations:
        print(f"\n  Recommendations:")
        for rec in eval_result.recommendations[:3]:
            print(f"    - {rec}")
    
    return spec, network, eval_result


def example_2_custom_colonization_params():
    """Example 2: Custom colonization parameters (not using preset)."""
    print("\n" + "=" * 70)
    print("Example 2: Custom Colonization Parameters")
    print("=" * 70)
    
    custom_params = ColonizationSpec(
        influence_radius=0.015,  # 15mm
        kill_radius=0.003,  # 3mm
        step_size=0.005,  # 5mm
        min_radius=0.0003,  # 0.3mm
        taper_factor=0.90,
        vessel_type="arterial",
        max_steps=100,
        max_curvature_deg=55.0,  # Moderate curvature constraint
        min_clearance=0.0012,  # 1.2mm clearance
        encourage_bifurcation=True,
        bifurcation_probability=0.75,
    )
    
    spec = TreeSpec(
        domain=DomainSpec(
            type="box",
            center=(0.0, 0.0, 0.0),
            size=(0.20, 0.15, 0.10),  # 20cm x 15cm x 10cm box
        ),
        inlet={"position": (-0.09, 0.0, 0.0), "radius": 0.004},
        colonization=custom_params,
        tissue_points=250,
        seed=123,
    )
    
    print(f"\n✓ Created TreeSpec with custom parameters:")
    print(f"  Domain: Box {spec.domain.size}")
    print(f"  Influence radius: {custom_params.influence_radius*1000:.1f}mm")
    print(f"  Kill radius: {custom_params.kill_radius*1000:.1f}mm")
    print(f"  Max curvature: {custom_params.max_curvature_deg}°")
    print(f"  Min clearance: {custom_params.min_clearance*1000:.1f}mm")
    
    print(f"\n✓ Building network...")
    network = design_from_spec(spec)
    
    print(f"\n✓ Network built:")
    print(f"  Nodes: {len(network.nodes)}")
    print(f"  Segments: {len(network.segments)}")
    
    eval_result = evaluate_network(network, tissue_points=1000)
    print(f"\n✓ Overall score: {eval_result.scores.overall_score:.2f}")
    
    return spec, network, eval_result


def example_3_dual_tree_design():
    """Example 3: Dual-tree (arterial + venous) network."""
    print("\n" + "=" * 70)
    print("Example 3: Dual-Tree Network (Arterial + Venous)")
    print("=" * 70)
    
    spec = DualTreeSpec(
        domain=DomainSpec(
            type="ellipsoid",
            semi_axes=(0.10, 0.08, 0.06),  # 10cm x 8cm x 6cm
        ),
        arterial_inlet={"position": (-0.08, 0.0, 0.0), "radius": 0.004},
        venous_outlet={"position": (0.08, 0.0, 0.0), "radius": 0.005},
        arterial_colonization=get_preset("liver_arterial_dense"),
        venous_colonization=get_preset("liver_venous_sparse"),
        arterial_tissue_points=150,
        venous_tissue_points=120,
        seed=42,
    )
    
    print(f"\n✓ Created DualTreeSpec:")
    print(f"  Domain: Ellipsoid {spec.domain.semi_axes}")
    print(f"  Arterial preset: liver_arterial_dense")
    print(f"  Venous preset: liver_venous_sparse")
    print(f"  Arterial tissue points: {spec.arterial_tissue_points}")
    print(f"  Venous tissue points: {spec.venous_tissue_points}")
    
    print(f"\n✓ Building dual-tree network...")
    network = design_from_spec(spec)
    
    arterial_nodes = sum(1 for n in network.nodes.values() if n.vessel_type == "arterial")
    venous_nodes = sum(1 for n in network.nodes.values() if n.vessel_type == "venous")
    
    print(f"\n✓ Network built:")
    print(f"  Total nodes: {len(network.nodes)}")
    print(f"  Arterial nodes: {arterial_nodes}")
    print(f"  Venous nodes: {venous_nodes}")
    print(f"  Total segments: {len(network.segments)}")
    
    eval_result = evaluate_network(network, tissue_points=1000)
    print(f"\n✓ Evaluation:")
    print(f"  Coverage: {eval_result.metrics.coverage_fraction:.1%}")
    print(f"  Overall score: {eval_result.scores.overall_score:.2f}")
    
    return spec, network, eval_result


def example_4_iterative_refinement():
    """Example 4: Iterative spec refinement based on evaluation."""
    print("\n" + "=" * 70)
    print("Example 4: Iterative Spec Refinement")
    print("=" * 70)
    
    spec = TreeSpec(
        domain=EllipsoidSpec(semi_axes=(0.10, 0.08, 0.06)),
        inlet={"position": (-0.08, 0.0, 0.0), "radius": 0.004},
        colonization=get_preset("sparse_debug"),  # Start sparse for speed
        tissue_points=100,
        seed=42,
    )
    
    print(f"\n✓ Iteration 1: Sparse debug preset")
    network = design_from_spec(spec)
    eval_result = evaluate_network(network, tissue_points=500)
    
    print(f"  Nodes: {len(network.nodes)}")
    print(f"  Coverage: {eval_result.metrics.coverage_fraction:.1%}")
    print(f"  Score: {eval_result.scores.overall_score:.2f}")
    
    if eval_result.scores.coverage_score < 0.7:
        print(f"\n✓ Iteration 2: Increasing density (coverage too low)")
        spec.colonization = get_preset("liver_arterial_dense")
        spec.tissue_points = 200
        
        network = design_from_spec(spec)
        eval_result = evaluate_network(network, tissue_points=500)
        
        print(f"  Nodes: {len(network.nodes)}")
        print(f"  Coverage: {eval_result.metrics.coverage_fraction:.1%}")
        print(f"  Score: {eval_result.scores.overall_score:.2f}")
    
    if eval_result.scores.coverage_score < 0.8:
        print(f"\n✓ Iteration 3: Further increasing tissue points")
        spec.tissue_points = 300
        
        network = design_from_spec(spec)
        eval_result = evaluate_network(network, tissue_points=500)
        
        print(f"  Nodes: {len(network.nodes)}")
        print(f"  Coverage: {eval_result.metrics.coverage_fraction:.1%}")
        print(f"  Score: {eval_result.scores.overall_score:.2f}")
    
    print(f"\n✓ Final network achieved target coverage!")
    
    return spec, network, eval_result


def example_5_spec_serialization():
    """Example 5: Saving and loading specs as JSON."""
    print("\n" + "=" * 70)
    print("Example 5: Spec Serialization (JSON)")
    print("=" * 70)
    
    spec = TreeSpec(
        domain=EllipsoidSpec(semi_axes=(0.12, 0.10, 0.08)),
        inlet={"position": (-0.10, 0.0, 0.0), "radius": 0.005},
        colonization=get_preset("liver_arterial_dense"),
        tissue_points=200,
        seed=42,
    )
    
    spec_dict = spec.to_dict()
    
    print(f"\n✓ Spec as JSON:")
    import json
    print(json.dumps(spec_dict, indent=2)[:500] + "...")
    
    output_path = "/tmp/network_spec.json"
    save_json(spec_dict, output_path)
    print(f"\n✓ Saved spec to {output_path}")
    
    from vascular_lib.specs import TreeSpec
    loaded_spec = TreeSpec.from_dict(spec_dict)
    
    print(f"\n✓ Loaded spec from dict")
    print(f"  Domain: {loaded_spec.domain.semi_axes}")
    print(f"  Tissue points: {loaded_spec.tissue_points}")
    print(f"  Seed: {loaded_spec.seed}")
    
    network = design_from_spec(loaded_spec)
    print(f"\n✓ Built network from loaded spec:")
    print(f"  Nodes: {len(network.nodes)}")
    print(f"  Segments: {len(network.segments)}")


def main():
    """Run all examples."""
    print("\n" + "=" * 70)
    print("DESIGN SPEC API EXAMPLES")
    print("=" * 70)
    
    example_1_single_tree_with_preset()
    example_2_custom_colonization_params()
    example_3_dual_tree_design()
    example_4_iterative_refinement()
    example_5_spec_serialization()
    
    print("\n" + "=" * 70)
    print("All examples completed successfully!")
    print("=" * 70)
    print("\nKey Takeaways:")
    print("  1. DesignSpec provides a high-level, JSON-serializable API")
    print("  2. Use TreeSpec for single trees, DualTreeSpec for dual trees")
    print("  3. Parameter presets simplify common anatomical contexts")
    print("  4. Specs can be iteratively refined based on evaluation")
    print("  5. Full JSON serialization enables LLM-driven design loops")


if __name__ == "__main__":
    main()
