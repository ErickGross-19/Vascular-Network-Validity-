import numpy as np
import matplotlib.pyplot as plt
import trimesh

from ..models import ValidationReport
from ..io.loaders import load_stl_mesh


def plot_surface_quality(mesh: trimesh.Trimesh, title: str) -> None:
    """
    Simple surface-quality visualization:
      - histogram of face areas
      - histogram of edge lengths
    """
    mesh = mesh.copy()

    areas = mesh.area_faces

    edges = mesh.edges_unique
    v = mesh.vertices
    lengths = np.linalg.norm(v[edges[:, 0]] - v[edges[:, 1]], axis=1)

    plt.figure(figsize=(6, 4))
    plt.hist(areas, bins=50)
    plt.xlabel("Face area")
    plt.ylabel("Count")
    plt.title(f"{title} - Face Area Distribution")
    plt.tight_layout()

    plt.figure(figsize=(6, 4))
    plt.hist(lengths, bins=50)
    plt.xlabel("Edge length")
    plt.ylabel("Count")
    plt.title(f"{title} - Edge Length Distribution")
    plt.tight_layout()


def plot_surface_quality_from_report(report: ValidationReport) -> None:
    """
    Load original and cleaned STLs from report and plot their surface-quality histograms.
    """
    mesh_orig = load_stl_mesh(report.input_file, process=False)
    mesh_clean = load_stl_mesh(report.cleaned_stl, process=False)

    plot_surface_quality(mesh_orig, "Original")
    plot_surface_quality(mesh_clean, "Cleaned")


def plot_volume_pipeline(report: ValidationReport) -> None:
    """
    Bar plot of volume across pipeline stages.
    """
    stages = ["Before", "Basic clean", "Voxel", "Repair"]
    vols = [
        report.before.volume,
        report.after_basic_clean.volume,
        report.after_voxel.volume,
        report.after_repair.volume,
    ]

    plt.figure(figsize=(6, 4))
    plt.bar(stages, vols)
    plt.ylabel("Volume")
    plt.title("Volume across pipeline stages")
    plt.tight_layout()


def plot_components_pipeline(report: ValidationReport) -> None:
    """
    Bar plot of component count across pipeline stages, plus watertight flags.
    """
    stages = ["Before", "Basic clean", "Voxel", "Repair"]
    comps = [
        report.before.num_components,
        report.after_basic_clean.num_components,
        report.after_voxel.num_components,
        report.after_repair.num_components,
    ]
    watertight = [
        report.before.watertight,
        report.after_basic_clean.watertight,
        report.after_voxel.watertight,
        report.after_repair.watertight,
    ]

    plt.figure(figsize=(6, 4))
    bars = plt.bar(stages, comps)
    for b, wt in zip(bars, watertight):
        color = "green" if wt else "red"
        b.set_edgecolor(color)
        b.set_linewidth(2.0)
    plt.ylabel("Number of connected components")
    plt.title("Components & watertightness across pipeline")
    plt.tight_layout()


def plot_connectivity_summary(report: ValidationReport) -> None:
    """
    Simple visualization of connectivity metrics from report.connectivity.
    """
    conn = report.connectivity or {}
    reachable_fraction = conn.get("reachable_fraction", None)
    num_components = conn.get("num_fluid_components", None)

    if reachable_fraction is not None:
        plt.figure(figsize=(4, 4))
        plt.bar(["reachable_fraction"], [reachable_fraction])
        plt.ylim(0, 1.0)
        plt.title("Connectivity reachable_fraction")
        plt.tight_layout()

    if num_components is not None:
        plt.figure(figsize=(4, 4))
        plt.bar(["components"], [num_components])
        plt.title("Number of fluid components")
        plt.tight_layout()


def plot_centerline_radii_summary(report: ValidationReport) -> None:
    """
    Bar plot of min/mean/max centerline radius from report.centerline_summary.
    """
    cls = report.centerline_summary or {}
    r_min = cls.get("radius_min", None)
    r_mean = cls.get("radius_mean", None)
    r_max = cls.get("radius_max", None)

    if r_min is None or r_mean is None or r_max is None:
        print("[plot_centerline_radii_summary] Radius summary not found.")
        return

    labels = ["min", "mean", "max"]
    values = [r_min, r_mean, r_max]

    plt.figure(figsize=(4, 4))
    plt.bar(labels, values)
    plt.ylabel("Radius")
    plt.title("Centerline radius summary")
    plt.tight_layout()
