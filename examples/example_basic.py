"""
Basic example of using the vascular network package.

This example demonstrates:
1. Loading an STL file
2. Running the validation and repair pipeline
3. Viewing the results
"""

from vascular_network import validate_and_repair_geometry
from vascular_network.visualization import show_full_report

print("Processing vascular network geometry...")

report, centerline_graph = validate_and_repair_geometry(
    input_path="input.stl",
    cleaned_stl_path="output_cleaned.stl",
    scaffold_stl_path="output_scaffold.stl",
    report_path="report.json",
    voxel_pitch=0.1,
    smooth_iters=40,
)

print("\n=== Validation Results ===")
print(f"Status: {report.flags.status}")
print(f"Watertight: {report.after_repair.watertight}")
print(f"Volume: {report.after_repair.volume}")
print(f"Number of components: {report.after_repair.num_components}")

show_full_report(report, centerline_graph)
