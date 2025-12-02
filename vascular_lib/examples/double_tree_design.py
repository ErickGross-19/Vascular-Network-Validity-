"""
Example: Designing a dual-tree vascular network with anastomoses.

This example demonstrates the full workflow for LLM-driven design of
arterial and venous trees with capillary connections.
"""

import numpy as np
from vascular_lib.core.domain import EllipsoidDomain
from vascular_lib.ops import (
    create_network,
    add_inlet,
    add_outlet,
    grow_branch,
    bifurcate,
    create_anastomosis,
    check_tree_interactions,
)
from vascular_lib.rules import (
    BranchingConstraints,
    RadiusRuleSpec,
    DegradationRuleSpec,
    InteractionRuleSpec,
)
from vascular_lib.analysis import (
    compute_perfusion_metrics,
    suggest_anastomosis_locations,
)
from vascular_lib.io import save_network_json


def main():
    """Build a dual-tree vascular network with anastomoses."""
    
    print("=" * 60)
    print("Double-Tree Vascular Network Design Example")
    print("=" * 60)
    
    print("\n1. Creating liver-like domain...")
    domain = EllipsoidDomain(
        center=(0, 0, 0),
        radii=(0.08, 0.06, 0.04),  # 80mm x 60mm x 40mm
    )
    
    result = create_network(domain, name="dual_tree_liver")
    network = result.metadata["network"]
    print(f"   ✓ Created network in {domain}")
    
    print("\n2. Setting up design rules...")
    
    branching_constraints = BranchingConstraints(
        min_radius=0.0001,  # 100 microns (capillary scale)
        max_radius=0.01,    # 10 mm
        max_branch_order=8,
        min_segment_length=0.002,
        max_segment_length=0.020,
        max_branch_angle_deg=60.0,
    )
    
    degradation_rule = DegradationRuleSpec.exponential(
        factor=0.85,  # 15% radius reduction per generation
        min_radius=0.0001,
    )
    
    interaction_rules = InteractionRuleSpec()  # Default: 1mm clearance, anastomosis allowed
    
    print(f"   ✓ Branching: min_radius={branching_constraints.min_radius*1000:.1f}mm")
    print(f"   ✓ Degradation: {degradation_rule.model}, factor={degradation_rule.degradation_factor}")
    print(f"   ✓ Interaction: clearance={interaction_rules.get_min_distance('arterial', 'venous')*1000:.1f}mm")
    
    print("\n3. Building arterial tree...")
    
    arterial_inlet = add_inlet(
        network,
        position=(-0.06, 0, 0),  # Left side
        direction=(1, 0, 0),     # Grow rightward
        radius=0.005,            # 5mm inlet
        vessel_type="arterial",
    )
    arterial_root = arterial_inlet.new_ids["node"]
    print(f"   ✓ Added arterial inlet at node {arterial_root}")
    
    arterial_main = grow_branch(
        network,
        from_node_id=arterial_root,
        length=0.025,
        direction=(1, 0, 0),
        target_radius=0.004,
        constraints=branching_constraints,
    )
    arterial_tip = arterial_main.new_ids["node"]
    print(f"   ✓ Grew main arterial branch to node {arterial_tip}")
    
    arterial_terminals = [arterial_tip]
    for gen in range(2):
        new_terminals = []
        for node_id in arterial_terminals:
            result = bifurcate(
                network,
                at_node_id=node_id,
                child_lengths=(0.015, 0.015),
                angle_deg=45.0,
                degradation_rule=degradation_rule,
                constraints=branching_constraints,
                seed=42 + gen,
            )
            if result.is_success():
                new_terminals.extend(result.new_ids["nodes"])
        arterial_terminals = new_terminals
        print(f"   ✓ Generation {gen+1}: {len(arterial_terminals)} arterial terminals")
    
    print("\n4. Building venous tree...")
    
    venous_outlet = add_outlet(
        network,
        position=(0.06, 0, 0),   # Right side
        direction=(-1, 0, 0),    # Grow leftward
        radius=0.006,            # 6mm outlet (slightly larger)
        vessel_type="venous",
    )
    venous_root = venous_outlet.new_ids["node"]
    print(f"   ✓ Added venous outlet at node {venous_root}")
    
    venous_main = grow_branch(
        network,
        from_node_id=venous_root,
        length=0.025,
        direction=(-1, 0, 0),
        target_radius=0.005,
        constraints=branching_constraints,
    )
    venous_tip = venous_main.new_ids["node"]
    print(f"   ✓ Grew main venous branch to node {venous_tip}")
    
    venous_terminals = [venous_tip]
    for gen in range(2):
        new_terminals = []
        for node_id in venous_terminals:
            result = bifurcate(
                network,
                at_node_id=node_id,
                child_lengths=(0.015, 0.015),
                angle_deg=45.0,
                degradation_rule=degradation_rule,
                constraints=branching_constraints,
                seed=100 + gen,
            )
            if result.is_success():
                new_terminals.extend(result.new_ids["nodes"])
        venous_terminals = new_terminals
        print(f"   ✓ Generation {gen+1}: {len(venous_terminals)} venous terminals")
    
    print("\n5. Checking tree interactions...")
    
    interaction_result = check_tree_interactions(network, rules=interaction_rules)
    
    violations = interaction_result.metadata["violations"]
    candidates = interaction_result.metadata["anastomosis_candidates"]
    stats = interaction_result.metadata["clearance_stats"]
    
    print(f"   ✓ Checked {stats['pairs_checked']} arterial-venous pairs")
    print(f"   ✓ Clearance violations: {len(violations)}")
    print(f"   ✓ Anastomosis candidates: {len(candidates)}")
    print(f"   ✓ Mean distance: {stats['mean_distance']*1000:.2f}mm")
    
    print("\n6. Analyzing tissue perfusion...")
    
    n_samples = 100
    tissue_points = []
    for _ in range(n_samples):
        theta = np.random.uniform(0, 2*np.pi)
        phi = np.random.uniform(0, np.pi)
        r = np.random.uniform(0, 1) ** (1/3)  # Uniform in volume
        
        x = 0.08 * r * np.sin(phi) * np.cos(theta)
        y = 0.06 * r * np.sin(phi) * np.sin(theta)
        z = 0.04 * r * np.cos(phi)
        tissue_points.append([x, y, z])
    
    tissue_points = np.array(tissue_points)
    
    perfusion_metrics = compute_perfusion_metrics(
        network,
        tissue_points,
        weights=(1.0, 1.0),  # Equal weight for arterial and venous
        distance_cap=0.020,  # 20mm max distance
    )
    
    print(f"   ✓ Analyzed {perfusion_metrics['total_points']} tissue points")
    print(f"   ✓ Well-perfused fraction: {perfusion_metrics['well_perfused_fraction']:.1%}")
    print(f"   ✓ Mean perfusion score: {perfusion_metrics['perfusion_stats']['mean']:.3f}")
    print(f"   ✓ Under-perfused regions: {len(perfusion_metrics['under_perfused_regions'])}")
    
    print("\n7. Creating anastomoses...")
    
    anastomosis_suggestions = suggest_anastomosis_locations(
        network,
        perfusion_metrics,
        rules=interaction_rules,
        k=5,  # Top 5 candidates
    )
    
    print(f"   ✓ Found {len(anastomosis_suggestions)} anastomosis suggestions")
    
    created_anastomoses = []
    for i, suggestion in enumerate(anastomosis_suggestions[:3]):  # Create top 3
        result = create_anastomosis(
            network,
            arterial_node_id=suggestion["arterial_node"],
            venous_node_id=suggestion["venous_node"],
            rules=interaction_rules,
        )
        
        if result.is_success():
            created_anastomoses.append(result.new_ids["segment"])
            print(f"   ✓ Anastomosis {i+1}: nodes {suggestion['arterial_node']} → {suggestion['venous_node']}, "
                  f"distance={suggestion['distance']*1000:.2f}mm")
        else:
            print(f"   ✗ Anastomosis {i+1} failed: {result.message}")
    
    print("\n8. Final network statistics:")
    print(f"   • Total nodes: {len(network.nodes)}")
    print(f"   • Total segments: {len(network.segments)}")
    print(f"   • Arterial nodes: {sum(1 for n in network.nodes.values() if n.vessel_type == 'arterial')}")
    print(f"   • Venous nodes: {sum(1 for n in network.nodes.values() if n.vessel_type == 'venous')}")
    print(f"   • Anastomoses: {len(created_anastomoses)}")
    
    print("\n9. Saving network...")
    output_path = "output/dual_tree_network.json"
    save_network_json(network, output_path)
    print(f"   ✓ Saved to {output_path}")
    
    print("\n" + "=" * 60)
    print("Double-tree network design complete!")
    print("=" * 60)
    
    return network


if __name__ == "__main__":
    network = main()
