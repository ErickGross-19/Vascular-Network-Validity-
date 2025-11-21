import json
from pathlib import Path
from typing import Dict, Any, Optional
import networkx as nx

from ..models import ValidationReport
from .mesh_plots import (
    plot_surface_quality_from_report,
    plot_volume_pipeline,
    plot_components_pipeline,
    plot_connectivity_summary,
    plot_centerline_radii_summary,
)
from .flow_plots import (
    plot_poiseuille_flows_from_report,
    plot_centerline_graph_2d,
    plot_centerline_scalar,
    plot_flow_distribution,
    plot_poiseuille_histograms,
    summarize_poiseuille,
)
from ..analysis.cfd import ensure_poiseuille_solved


def load_report_json(json_path: str | Path) -> Dict[str, Any]:
    """
    Load a saved JSON report.
    """
    json_path = Path(json_path)
    with open(json_path, "r", encoding="utf-8") as f:
        return json.load(f)


def pretty_print_validation_report(report: ValidationReport) -> None:
    """
    Human-readable summary of a ValidationReport object.
    """
    print("=== Validation Report ===")
    print(f"Input file       : {report.input_file}")
    print(f"Intermediate STL : {report.intermediate_stl}")
    print(f"Cleaned STL      : {report.cleaned_stl}")
    print(f"Scaffold STL      : {report.scafold_stl}")
    print()

    print("--- Status & Flags ---")
    print(f"Status : {report.flags.status}")
    if report.flags.flags:
        for f in report.flags.flags:
            print(f"  - {f}")
    else:
        print("  (no flags)")
    print()

    print("--- Diagnostics (volumes & components) ---")
    print("Before           :",
          "Volume=", report.before.volume,
          "components=", report.before.num_components,
          "watertight=", report.before.watertight,
          "source=", report.before.volume_source,
          "bounding box=", report.before.bounding_box_extents)

    print("After basic clean:",
          "Volume=", report.after_basic_clean.volume,
          "components=", report.after_basic_clean.num_components,
          "watertight=", report.after_basic_clean.watertight,
          "source=", report.after_basic_clean.volume_source,
          "bounding box=", report.after_basic_clean.bounding_box_extents)

    print("After voxel      :",
          "Volume=", report.after_voxel.volume,
          "components=", report.after_voxel.num_components,
          "watertight=", report.after_voxel.watertight,
          "source=", report.after_voxel.volume_source,
          "bounding box=", report.after_voxel.bounding_box_extents)

    print("After repair     :",
          "Volume=", report.after_repair.volume,
          "components=", report.after_repair.num_components,
          "watertight=", report.after_repair.watertight,
          "source=", report.after_repair.volume_source,
          "bounding box=", report.after_repair.bounding_box_extents)
    print()

    print("--- Connectivity ---")
    conn = report.connectivity or {}
    for k, v in conn.items():
        print(f"{k}: {v}")
    print()

    print("--- Centerline summary ---")
    cls = report.centerline_summary or {}
    meta = cls.get("meta", {})
    print(f"Centerline nodes/edges: {meta.get('num_nodes', 'N/A')} / "
          f"{meta.get('num_edges', 'N/A')}")
    print(f"Radii (min/mean/max): "
          f"{cls.get('radius_min', 'N/A')} / "
          f"{cls.get('radius_mean', 'N/A')} / "
          f"{cls.get('radius_max', 'N/A')}")
    print()

    print("--- Poiseuille summary ---")
    poi = report.poiseuille_summary or {}
    tin = poi.get("total_inlet_flow", poi.get("total_inflow", None))
    tout = poi.get("total_outlet_flow", poi.get("total_outflow", None))
    print(f"Num nodes        : {poi.get('num_nodes', 'N/A')}")
    print(f"Num edges        : {poi.get('num_edges', 'N/A')}")
    print(f"Inlets/Outlets   : {poi.get('num_inlets', 'N/A')} / {poi.get('num_outlets', 'N/A')}")
    print(f"Interior nodes   : {poi.get('num_interior', 'N/A')}")
    print(f"Matrix shape     : {poi.get('matrix_shape', 'N/A')}")
    print(f"Matrix rank      : {poi.get('matrix_rank', 'N/A')}")
    print(f"Used lstsq       : {poi.get('used_lstsq', 'N/A')}")
    print(f"warnings       : {poi.get('warnings', 'N/A')}")
    if tin is not None and tout is not None:
        print(f"Total inlet flow : {tin:.6g}")
        print(f"Total outlet flow: {tout:.6g}")
    print()


def show_full_report(
    report: ValidationReport, G_centerline: Optional[nx.Graph] = None,
) -> None:
    """
    Convenience function:
      - Print a textual summary
      - Show pipeline diagnostic plots
      - Surface-quality histograms
      - Connectivity, centerline, and Poiseuille summaries
      - Optionally draw a 2D projection of the centerline graph
    """
    pretty_print_validation_report(report)

    plot_volume_pipeline(report)
    plot_components_pipeline(report)

    plot_surface_quality_from_report(report)

    plot_connectivity_summary(report)

    plot_centerline_radii_summary(report)

    plot_poiseuille_flows_from_report(report)

    if G_centerline is not None:
        plot_centerline_graph_2d(
            G_centerline, plane="xz",
            title="Centerline graph (XZ projection)"
        )

        G_poi = ensure_poiseuille_solved(G_centerline, delta_p=1000.0, mu=1.0e-3)

        summarize_poiseuille(G_poi)
        plot_centerline_scalar(G_poi, edge_attr="Q", log_abs=True,
                               title="Flow magnitude")
        plot_flow_distribution(G_poi, edge_attr="Q")
        plot_poiseuille_histograms(G_poi)
