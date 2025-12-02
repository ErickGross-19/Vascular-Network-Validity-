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
from vascular_lib.core import EllipsoidDomain

# Create liver-shaped domain
domain = EllipsoidDomain(0.12, 0.10, 0.08)  # 12cm x 10cm x 8cm

# Create network
network = create_network(domain, seed=42)

# Add inlet
result = add_inlet(
    network,
    position=(-0.10, 0.0, 0.0),
    direction=(1.0, 0.0, 0.0),
    radius=0.005,
)
inlet_id = result.new_ids['node']

# Grow branch
result = grow_branch(
    network,
    from_node_id=inlet_id,
    length=0.02,
    direction=(1.0, 0.0, 0.0),
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
- `estimate_flows()` - Hemodynamic flow estimation (simplified)
- `solve_flow()` - Full network flow solver using scipy.sparse
- `check_flow_plausibility()` - Validate hemodynamic plausibility
- `check_hemodynamic_plausibility()` - Validate biophysical rules

**Collision:**
- `get_collisions()` - Find segment collisions
- `avoid_collisions()` - Collision detection/repair with strategies:
  - `"report"` - Just report collisions
  - `"reroute"` - Attempt to reroute branches (cone search)
  - `"shrink"` - Reduce radius/length
  - `"terminate"` - Mark branches as terminal

**Export & Integration:**
- `to_trimesh()` - Export to STL mesh (fast/robust modes)
- `export_stl()` - Export to STL file with repair
- `to_networkx_graph()` - Convert to NetworkX for analysis
- `from_networkx_graph()` - Convert from NetworkX
- `make_full_report()` - Comprehensive LLM-friendly report

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

### LLM Design Loop

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

### Full Pipeline

See `vascular_lib/examples/full_pipeline.py` for end-to-end workflow:

```bash
python -m vascular_lib.examples.full_pipeline
```

Pipeline demonstrates:
1. Create domain and network
2. Add inlet/outlet nodes
3. Grow arterial tree with space colonization
4. Grow venous tree
5. Check and repair collisions
6. Solve hemodynamics with full solver
7. Export STL mesh with repair
8. Generate comprehensive report

## Architecture

```
vascular_lib/
├── core/           # Data structures (Point3D, Node, VascularNetwork, etc.)
├── rules/          # Constraints and biophysical rules
├── spatial/        # Spatial indexing for collision detection
├── ops/            # Operations (build, growth, collision, space_colonization)
├── analysis/       # Query, analysis, and full flow solver
├── adapters/       # Integration with existing vascular_network package
├── io/             # JSON serialization
├── tests/          # Unit tests
└── examples/       # Example scripts
```

## Design Principles

1. **Small operations** - Each function does one thing
2. **Explicit state** - No hidden globals or side effects
3. **Structured results** - OperationResult with status, IDs, warnings, errors, error_codes
4. **Serializability** - Everything has to_dict()/from_dict(), JSON-safe
5. **Deterministic** - Accept seed/rng, return rng_state
6. **Reversible** - Delta objects for undo/redo
7. **LLM-friendly** - Error codes, structured feedback, explicit parameters

## Biophysical Rules

- **Murray's Law** - Radius scaling at bifurcations (r³ conservation)
- **Poiseuille Flow** - Resistance = 8μL/(πr⁴)
- **Reynolds Number** - Laminar flow check (Re < 2300)
- **Collision Avoidance** - Minimum clearance between vessels
- **Domain Constraints** - Growth bounded by organ geometry

## Features

### Implemented
- ✓ Core data model with full JSON serialization
- ✓ Construction operations (create_network, add_inlet, add_outlet)
- ✓ Growth operations (grow_branch, bifurcate)
- ✓ Space colonization for organic growth
- ✓ Collision detection and repair (reroute, shrink, terminate)
- ✓ Full network flow solver using scipy.sparse
- ✓ STL mesh export with repair pipeline
- ✓ Integration with existing vascular_network package
- ✓ Comprehensive unit tests
- ✓ Undo/redo support
- ✓ Error codes for LLM consumption

### Future Extensions
- dry_run parameter for all mutating operations
- Multi-tree anastomosis connections
- Optimization objectives (minimize resistance, maximize coverage)
- Additional organ-specific rule sets (kidney, lung, brain)
- Pydantic models for JSON schema validation

## License

See repository LICENSE file.
