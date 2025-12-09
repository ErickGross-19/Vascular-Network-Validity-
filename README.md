# Vascular Network Tools

A comprehensive suite of Python tools for designing, generating, and validating vascular networks. This repository contains three main components that work together to support the full lifecycle of vascular network development.

## 🎯 Choose Your Path

### 🧠 Design Networks with AI (vascular_lib)
**For:** LLM-driven iterative design, programmatic network construction  
**Use when:** You want to design custom vascular networks using composable operations

```python
from vascular_lib import create_network, add_inlet, grow_branch, bifurcate
from vascular_lib.core import EllipsoidDomain

domain = EllipsoidDomain(semi_axis_a=0.12, semi_axis_b=0.1, semi_axis_c=0.08)
network = create_network(domain, seed=42)
result = add_inlet(network, position=(-0.10, 0, 0), direction=(1, 0, 0), radius=0.005)
```

👉 [Learn more about vascular_lib](vascular_lib/README.md)

### 🌳 Generate Organ Networks (generators)
**For:** Creating realistic organ-specific vascular networks  
**Use when:** You need a complete arterial/venous tree for a specific organ

```python
from generators.liver import generate_liver_network

network = generate_liver_network(
    arterial_segments=500,
    venous_segments=500,
    output_dir="output/",
)
```

👉 [Learn more about generators](generators/README.md)

### ✅ Validate & Analyze (vascular_network)
**For:** Mesh repair, CFD analysis, quality validation  
**Use when:** You have an STL file or network that needs validation and analysis

```python
from vascular_network import validate_and_repair_geometry

report, centerline = validate_and_repair_geometry(
    input_path="network.stl",
    cleaned_stl_path="cleaned.stl",
    report_path="report.json",
)
```

👉 [Learn more about vascular_network](vascular_network/README.md)

## 📦 Installation

```bash
git clone https://github.com/ErickGross-19/Vascular-Network-Validity-.git
cd Vascular-Network-Validity-
pip install -e .
```

### Dependencies

Core dependencies:
- numpy >= 1.20.0
- trimesh >= 3.9.0
- networkx >= 2.5
- scipy >= 1.6.0

Additional dependencies for specific components:
- pymeshfix >= 0.16.0 (for mesh repair)
- pyvista >= 0.32.0 (for visualization)
- scikit-image >= 0.18.0 (for voxel processing)
- matplotlib >= 3.3.0 (for plotting)

## 🚀 Quick Start Examples

### End-to-End Workflow (High-Level API - Recommended)

```python
# 1. Design a network with vascular_lib using DesignSpec
from vascular_lib.api import design_from_spec, evaluate_network, run_experiment
from vascular_lib.specs import TreeSpec, DomainSpec
from vascular_lib.params import get_preset

# Define network design as JSON spec
spec = TreeSpec(
    domain=DomainSpec.ellipsoid(semi_axes=(0.12, 0.10, 0.08)),
    inlet={"position": (-0.10, 0.0, 0.0), "radius": 0.005},
    colonization=get_preset("liver_arterial_dense"),
    tissue_points=200,
    seed=42,
)

# Build network from spec
network = design_from_spec(spec)

# Evaluate network quality
eval_result = evaluate_network(network, tissue_points=1000)
print(f"Coverage: {eval_result.metrics.coverage_fraction:.1%}")
print(f"Overall score: {eval_result.scores.overall_score:.2f}")

# Or run complete experiment with one call
result = run_experiment(spec, output_dir="output/my_network")
# Generates: network.json, network.stl, eval_report.json, spec.json
```

### End-to-End Workflow (Low-Level API)

```python
# 1. Design a network with vascular_lib
from vascular_lib import create_network, add_inlet, space_colonization_step
from vascular_lib.core import EllipsoidDomain
from vascular_lib.params import get_preset
import numpy as np

domain = EllipsoidDomain(semi_axis_a=0.12, semi_axis_b=0.1, semi_axis_c=0.08)
network = create_network(domain, seed=42)

# Add inlet
add_inlet(network, position=(-0.10, 0, 0), direction=(1, 0, 0), radius=0.005)

# Use space colonization with preset parameters
tissue_points = domain.sample_points(n_points=200, seed=42)
params = get_preset("liver_arterial_dense")
space_colonization_step(network, tissue_points, params, seed=42)

# 2. Export to STL
from vascular_lib.adapters import export_stl, export_hollow_tube_stl
export_stl(network, output_path="network.stl", mode="fast", repair=True)

# Or export as hollow tube for fluid flow
export_hollow_tube_stl(network, output_path="hollow_network.stl", wall_thickness=1.0)

# 3. Validate and analyze with vascular_network
from vascular_network import validate_and_repair_geometry
report, centerline = validate_and_repair_geometry(
    input_path="network.stl",
    cleaned_stl_path="cleaned.stl",
    report_path="report.json",
)

print(f"Watertight: {report.after_repair.watertight}")
print(f"Volume: {report.after_repair.volume:.6f} m³")
```

### Generate a Liver Network

```python
from generators.liver import generate_liver_network

# Generate complete liver vascular network
network = generate_liver_network(
    arterial_segments=500,
    venous_segments=500,
    output_dir="output/liver_network",
    seed=42,
)

# Outputs:
# - output/liver_network/arterial_tree.py
# - output/liver_network/venous_tree.py
# - output/liver_network/network_metadata.json
```

## 🏗️ Repository Structure

```
Vascular-Network-Validity-/
├── vascular_lib/          # LLM-friendly design library
│   ├── core/              # Data structures (Node, Segment, Network)
│   ├── ops/               # Operations (grow, bifurcate, colonize)
│   ├── analysis/          # Flow solver, coverage analysis
│   ├── adapters/          # Integration with vascular_network
│   ├── examples/          # Example scripts
│   └── tests/             # Unit tests
│
├── generators/            # Organ-specific network generators
│   └── liver/             # Liver vascular network generator
│       ├── config.py      # Configuration parameters
│       ├── geometry.py    # Liver domain geometry
│       ├── growth.py      # Growth algorithms
│       └── export.py      # Export to Python/JSON
│
├── vascular_network/      # Validation and analysis tools
│   ├── io/                # STL/Python file loaders
│   ├── mesh/              # Mesh repair and diagnostics
│   ├── analysis/          # Centerline, CFD, metrics
│   ├── visualization/     # Plotting functions
│   ├── examples/          # Example scripts
│   └── tests/             # Unit tests
│
└── docs/                  # Documentation
    └── legacy/            # Archived legacy code
```

## 🧪 Testing

Run all tests:
```bash
pytest vascular_lib/tests/
pytest vascular_network/tests/
```

Run specific test suites:
```bash
# Test LLM library
pytest vascular_lib/tests/ -v

# Test validation package
pytest vascular_network/tests/ -v

# Test with coverage
pytest --cov=vascular_lib --cov=vascular_network
```

## 📚 Documentation

- [vascular_lib Documentation](vascular_lib/README.md) - LLM-friendly design library
- [generators Documentation](generators/README.md) - Organ-specific network generators
- [vascular_network Documentation](vascular_network/README.md) - Validation and analysis tools

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🔗 Related Work

- [vascular_lib](vascular_lib/) - Composable operations for LLM-driven design
- [generators](generators/) - Organ-specific network generation
- [vascular_network](vascular_network/) - Validation, repair, and CFD analysis

## 📖 Citation

If you use this package in your research, please cite:

```bibtex
@software{vascular_network_tools,
  author = {Gross, Erick},
  title = {Vascular Network Tools: Design, Generation, and Validation},
  year = {2025},
  url = {https://github.com/ErickGross-19/Vascular-Network-Validity-}
}
```

## 📧 Contact

For questions or issues, please open an issue on GitHub or contact:
- Erick Gross - erickgross1924@gmail.com
