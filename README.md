# Vascular Network Validation and Analysis Package

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

### From Source

```bash
git clone https://github.com/ErickGross-19/Vascular-Network-Validity-.git
cd Vascular-Network-Validity-
pip install -e .
```

### Dependencies

The package requires the following dependencies:
- numpy >= 1.20.0
- trimesh >= 3.9.0
- pymeshfix >= 0.16.0
- pyvista >= 0.32.0
- networkx >= 2.5
- scipy >= 1.6.0
- scikit-image >= 0.18.0
- matplotlib >= 3.3.0
- cadquery >= 2.0

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

See the `examples/` directory for complete usage examples:
- `example_basic.py`: Basic usage with STL files
- `example_cad.py`: Processing Python CAD files
- `example_analysis.py`: Advanced analysis and visualization

## Testing

Run the test suite:

```bash
pytest tests/
```

The test suite includes:
- Unit tests for cylinder geometries
- Unit tests for Y-branch (branching) structures
- Tests for mesh operations and analysis functions

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Citation

If you use this package in your research, please cite:

```
@software{vascular_network,
  author = {Gross, Erick},
  title = {Vascular Network Validation and Analysis Package},
  year = {2025},
  url = {https://github.com/ErickGross-19/Vascular-Network-Validity-}
}
```

## Contact

For questions or issues, please open an issue on GitHub or contact:
- Erick Gross - erickgross1924@gmail.com
