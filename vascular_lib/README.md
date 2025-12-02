# Vascular Design Library

A composable "LEGO kit" library for LLM-driven iterative vascular network design.

## Overview

This library provides a clean, explicit API for designing vascular networks through small, composable operations. It's specifically designed for LLM consumption with:

- **Small, composable operations** - Each function does one thing well
- **Structured feedback** - Every operation returns explicit status, warnings, and errors
- **Full serializability** - Everything converts to/from JSON for LLM reasoning
- **Deterministic randomness** - Accept seeds, return RNG state for replayability
- **Reversible changes** - Delta/undo support for iterative design

## Installation

```bash
# From the repository root
pip install -e .
```

## Quick Start

```python
from vascular_lib import (
    create_network,
    add_inlet,
    grow_branch,
    bifurcate,
    save_json,
)
from vascular_lib.core import EllipsoidDomain, Direction3D

# Create liver-shaped domain
domain = EllipsoidDomain(semi_axis_a=0.12, semi_axis_b=0.10, semi_axis_c=0.08)

# Create network
network = create_network(domain, seed=42)

# Add inlet
result = add_inlet(
    network,
    position=(-0.10, 0.0, 0.0),
    direction=Direction3D(dx=1.0, dy=0.0, dz=0.0),
    radius=0.005,
)
inlet_id = result.new_ids['node']

# Grow branch
result = grow_branch(
    network,
    from_node_id=inlet_id,
    length=0.02,
    direction=Direction3D(dx=1.0, dy=0.0, dz=0.0),
)

# Create bifurcation
result = bifurcate(
    network,
    at_node_id=result.new_ids['node'],
    child_lengths=(0.015, 0.015),
    angle_deg=45.0,
)

# Save to JSON
save_json(network, "my_network.json")
```

## Core Concepts

### Data Model

- **Point3D** - 3D position
- **Direction3D** - Unit vector
- **TubeGeometry** - Tubular segment with start/end points and radii
- **Node** - Junction, inlet, outlet, or terminal point
- **VesselSegment** - Tube connecting two nodes
- **VascularNetwork** - Complete network with nodes, segments, and domain

### Domains

- **EllipsoidDomain** - Ellipsoidal organ shape (e.g., liver)
- **MeshDomain** - STL mesh-based domain

### Operations

**Construction:**
- `create_network()` - Create empty network
- `add_inlet()` - Add inlet node
- `add_outlet()` - Add outlet node

**Growth:**
- `grow_branch()` - Extend branch from node
- `bifurcate()` - Create bifurcation with Murray's law
- `space_colonization_step()` - Organic space-filling growth

**Analysis:**
- `get_leaf_nodes()` - Find terminal nodes
- `get_paths_from_inlet()` - Trace paths through network
- `compute_coverage()` - Tissue perfusion analysis
- `estimate_flows()` - Hemodynamic flow estimation
- `check_hemodynamic_plausibility()` - Validate biophysical rules

**Collision:**
- `get_collisions()` - Find segment collisions
- `avoid_collisions()` - Collision detection/repair

**I/O:**
- `save_json()` - Save network to JSON
- `load_json()` - Load network from JSON

### Rules and Constraints

**BranchingConstraints:**
- Min/max radius, segment length, branch order
- Max branching angle, curvature limit
- Termination rules

**RadiusRuleSpec:**
- Murray's law (r_parent³ = Σr_children³)
- Fixed radius
- Linear taper

**InteractionRuleSpec:**
- Minimum clearance between vessel types
- Anastomosis rules
- Parallel preference

## LLM Design Loop

The typical LLM-driven design loop:

```python
# 1. Create network
network = create_network(domain, seed=42)

# 2. Add boundary conditions
add_inlet(network, ...)
add_outlet(network, ...)

# 3. Query current state
coverage = compute_coverage(network, tissue_points)
leaf_nodes = get_leaf_nodes(network)

# 4. Take design action based on feedback
if coverage['fraction_covered'] < 0.8:
    # Grow towards uncovered regions
    for region in coverage['uncovered_regions']:
        nearest_node = region['nearest_node']
        grow_branch(network, from_node_id=nearest_node, ...)

# 5. Check results
collisions = get_collisions(network)
warnings = check_hemodynamic_plausibility(network)

# 6. Iterate until satisfied
```

## Examples

See `vascular_lib/examples/llm_design_loop.py` for complete examples:

```bash
python -m vascular_lib.examples.llm_design_loop
```

Examples include:
1. Basic network construction
2. Manual growth and bifurcation
3. Space colonization
4. Coverage analysis
5. Flow analysis
6. Serialization
7. Query operations

## Architecture

```
vascular_lib/
├── core/           # Data structures (Point3D, Node, VascularNetwork, etc.)
├── rules/          # Constraints and biophysical rules
├── spatial/        # Spatial indexing for collision detection
├── ops/            # Operations (build, growth, collision, space_colonization)
├── analysis/       # Query and analysis functions
├── io/             # JSON serialization
└── examples/       # Example scripts
```

## Design Principles

1. **Small operations** - Each function does one thing
2. **Explicit state** - No hidden globals or side effects
3. **Structured results** - OperationResult with status, IDs, warnings, errors
4. **Serializability** - Everything has to_dict()/from_dict()
5. **Deterministic** - Accept seed/rng, return rng_state
6. **Reversible** - Delta objects for undo/redo

## Biophysical Rules

- **Murray's Law** - Radius scaling at bifurcations (r³ conservation)
- **Poiseuille Flow** - Resistance = 8μL/(πr⁴)
- **Reynolds Number** - Laminar flow check (Re < 2300)
- **Collision Avoidance** - Minimum clearance between vessels
- **Domain Constraints** - Growth bounded by organ geometry

## Future Extensions

- Advanced collision repair strategies
- Multi-tree anastomosis connections
- CFD integration for detailed flow
- Optimization objectives (minimize resistance, maximize coverage)
- Additional organ-specific rule sets (kidney, lung, brain)

## License

See repository LICENSE file.
