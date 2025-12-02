# Vascular Network Validation and Analysis

A comprehensive Python package for validating, repairing, and analyzing vascular network geometries. This package provides tools for processing STL and Python CAD files, generating watertight meshes, performing connectivity analysis, extracting centerlines, and running CFD (Poiseuille flow) simulations.

## Features

- **Input Processing**: Support for STL and Python CAD files (CadQuery)
- **Mesh Repair**: Automated mesh cleaning, voxel remeshing, and watertight generation
- **Analysis Tools**:
  - Mesh diagnostics (watertightness, Euler number, components)
  - Surface quality metrics (face areas, edge lengths, aspect ratios)
  - Connectivity analysis (voxel-based)
  - Centerline extraction from voxelized fluid domains
  - Poiseuille flow network analysis (CFD)
- **Visualization**: Comprehensive plotting functions for mesh quality, flow distribution, and analysis results
- **Reporting**: JSON and text-based validation reports with all metrics

## Installation

From the repository root:

```bash
pip install -e .
```

## Quick Start

### Basic Usage

```python
from vascular_network import validate_and_repair_geometry

# Process an STL file
report, centerline_graph = validate_and_repair_geometry(
    input_path="input.stl",
    cleaned_stl_path="output_cleaned.stl",
    scaffold_stl_path="output_scaffold.stl",
    report_path="report.json",
    voxel_pitch=0.1,
    smooth_iters=40,
)

# Check the validation status
print(f"Status: {report.flags.status}")
print(f"Watertight: {report.after_repair.watertight}")
print(f"Volume: {report.after_repair.volume}")
```

### Using Individual Functions

```python
from vascular_network.io import load_stl_mesh
from vascular_network.mesh import basic_clean, voxel_remesh_and_smooth
from vascular_network.analysis import extract_centerline_graph, compute_poiseuille_network
from vascular_network.visualization import show_full_report

# Load and clean mesh
mesh = load_stl_mesh("input.stl")
mesh_clean = basic_clean(mesh)
mesh_voxel = voxel_remesh_and_smooth(mesh_clean, pitch=0.1)

# Analyze connectivity and extract centerline
from vascular_network.analysis import analyze_connectivity_voxel
connectivity_info, fluid_mask, origin, pitch = analyze_connectivity_voxel(mesh_voxel)
centerline_graph, meta = extract_centerline_graph(fluid_mask, origin, pitch)

# Run CFD analysis
cfd_results = compute_poiseuille_network(centerline_graph, mu=1.0)

# Visualize results
show_full_report(report, centerline_graph)
```

### Processing Python CAD Files

```python
# Process a CadQuery Python file
report, centerline_graph = validate_and_repair_geometry(
    input_path="model.py",  # CadQuery script
    cleaned_stl_path="output_cleaned.stl",
    report_path="report.json",
)
```

## API Reference

### Main Pipeline Function

#### `validate_and_repair_geometry(input_path, ...)`

Full validation and repair pipeline that processes input files and generates comprehensive reports.

**Parameters:**
- `input_path` (str or Path): Path to input STL or Python CAD file
- `cleaned_stl_path` (str or Path, optional): Path to save cleaned STL
- `scaffold_stl_path` (str or Path, optional): Path to save scaffold shell STL
- `wall_thickness` (float): Wall thickness for scaffold shell (default: 0.4)
- `report_path` (str or Path, optional): Path to save JSON report
- `voxel_pitch` (float): Voxel pitch for remeshing (default: 0.1)
- `smooth_iters` (int): Number of smoothing iterations (default: 40)
- `dilation_iters` (int): Number of dilation iterations (default: 2)
- `inlet_nodes` (sequence, optional): Inlet nodes for CFD analysis
- `outlet_nodes` (sequence, optional): Outlet nodes for CFD analysis

**Returns:**
- `report` (ValidationReport): Validation report with all metrics
- `centerline_graph` (nx.Graph): Centerline graph with CFD results

### Module Organization

- `vascular_network.io`: Input/output operations (loaders, exporters)
- `vascular_network.mesh`: Mesh processing (cleaning, repair, diagnostics)
- `vascular_network.analysis`: Analysis operations (connectivity, centerline, CFD)
- `vascular_network.visualization`: Plotting and visualization functions
- `vascular_network.models`: Data classes for reports and metrics

## Examples

See the `vascular_network/examples/` directory for complete usage examples:
- `example_basic.py`: Basic usage with STL files
- `example_advanced.py`: Advanced analysis and visualization
- `generate_liver_network.py`: Generate liver vascular networks

Run examples:
```bash
python vascular_network/examples/example_basic.py
python vascular_network/examples/example_advanced.py
```

## Testing

Run the test suite:

```bash
pytest vascular_network/tests/
```

The test suite includes:
- Unit tests for cylinder geometries
- Unit tests for Y-branch (branching) structures
- Tests for mesh operations and analysis functions

## Advanced Features

### LLM Context Reports

Generate comprehensive reports for LLM consumption:

```python
from vascular_network.visualization import make_llm_context_report, save_llm_context_report_json

# Generate report
report_dict = make_llm_context_report(
    mesh=mesh,
    centerline_graph=centerline_graph,
    validation_report=report,
)

# Save to JSON
save_llm_context_report_json(report_dict, "llm_report.json")
```

### Advanced Visualization

```python
from vascular_network.visualization import (
    plot_mesh_quality_clean,
    plot_flow_distribution_clean,
    plot_centerline_3d_clean,
)

# Plot mesh quality
plot_mesh_quality_clean(mesh, report)

# Plot flow distribution
plot_flow_distribution_clean(centerline_graph)

# Plot 3D centerline
plot_centerline_3d_clean(centerline_graph, mesh)
```

## Module Details

### IO Module (`vascular_network.io`)

- `load_stl_mesh(path)` - Load STL file
- `load_python_cad(path)` - Load and execute CadQuery Python file
- `export_stl(mesh, path)` - Export mesh to STL

### Mesh Module (`vascular_network.mesh`)

- `basic_clean(mesh)` - Remove duplicates, degenerate faces
- `voxel_remesh_and_smooth(mesh, pitch)` - Voxelize and smooth
- `compute_diagnostics(mesh)` - Compute mesh quality metrics
- `repair_mesh(mesh)` - Repair non-watertight meshes

### Analysis Module (`vascular_network.analysis`)

- `analyze_connectivity_voxel(mesh)` - Voxel-based connectivity
- `extract_centerline_graph(fluid_mask, origin, pitch)` - Extract centerline
- `compute_poiseuille_network(graph, mu, inlet_nodes, outlet_nodes)` - CFD solver
- `compute_geometry_metrics(graph)` - Geometric analysis
- `compute_flow_metrics(graph)` - Flow analysis

### Visualization Module (`vascular_network.visualization`)

- `show_full_report(report, graph)` - Display complete report
- `plot_mesh_quality_clean(mesh, report)` - Mesh quality plots
- `plot_flow_distribution_clean(graph)` - Flow distribution plots
- `plot_centerline_3d_clean(graph, mesh)` - 3D centerline visualization

## Dependencies

- numpy >= 1.20.0
- trimesh >= 3.9.0
- pymeshfix >= 0.16.0
- pyvista >= 0.32.0
- networkx >= 2.5
- scipy >= 1.6.0
- scikit-image >= 0.18.0
- matplotlib >= 3.3.0

## License

See repository LICENSE file.

## Contact

For questions or issues, please open an issue on GitHub or contact:
- Erick Gross - erickgross1924@gmail.com
