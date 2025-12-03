import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from matplotlib.collections import LineCollection
from math import pi
from typing import Optional

from ..models import ValidationReport
from ..analysis.cfd import ensure_poiseuille_solved


def plot_poiseuille_flows_from_report(report: ValidationReport) -> None:
    """
    Bar plot of total inlet vs outlet flow, based on report.poiseuille_summary.
    """
    poi = report.poiseuille_summary or {}
    tin = poi.get("total_inlet_flow", poi.get("total_inflow", None))
    tout = poi.get("total_outlet_flow", poi.get("total_outflow", None))

    if tin is None or tout is None:
        print("[plot_poiseuille_flows_from_report] Flow summary not found.")
        return

    plt.figure(figsize=(4, 4))
    plt.bar(["Inlet", "Outlet"], [tin, tout])
    plt.ylabel("Flow")
    plt.title("Total inlet vs outlet flow")
    plt.tight_layout()


def plot_centerline_graph_2d(
    G,
    plane="xz",
    title="Centerline graph",
    size_by="junction",
    base_px=3.0,
    radius_scale=2000.0,
    min_px=2.0,
    max_px=50.0,
):
    """
    Plot a 2D projection of the centerline graph with node sizes scaled by junction properties.
    
    Parameters
    ----------
    G : networkx.Graph
        Centerline graph
    plane : str
        Projection plane: 'xz', 'xy', or 'yz'
    title : str
        Plot title
    size_by : str
        Node sizing method: 'junction' (Murray + degree), 'max_radius', 'mean_radius', 'degree', 'fixed'
    base_px : float
        Base node size in points^2
    radius_scale : float
        Scaling factor for radius contribution
    min_px : float
        Minimum node size
    max_px : float
        Maximum node size
    """
    from ..analysis.node_metrics import compute_node_display_sizes
    
    plt.figure(figsize=(4, 6))
    xs, ys = [], []
    node_ids = []

    for n, data in G.nodes(data=True):
        p = np.asarray(data.get("pos", data.get("coord", [0, 0, 0])), dtype=float)
        if plane == "xy":
            xs.append(p[0]); ys.append(p[1])
        elif plane == "yz":
            xs.append(p[1]); ys.append(p[2])
        else:
            xs.append(p[0]); ys.append(p[2])
        node_ids.append(n)

    if size_by == 'fixed':
        sizes = [5.0] * len(node_ids)
    else:
        size_dict = compute_node_display_sizes(
            G,
            size_by=size_by,
            base_px=base_px,
            radius_scale=radius_scale,
            min_px=min_px,
            max_px=max_px,
        )
        sizes = [size_dict.get(nid, 5.0) for nid in node_ids]

    plt.scatter(xs, ys, s=sizes)
    plt.xlabel(plane[0])
    plt.ylabel(plane[1])
    plt.title(title)
    plt.axis("equal")
    plt.tight_layout()
    plt.show()


def plot_centerline_scalar(
    G,
    edge_attr="Q",
    cmap="viridis",
    log_abs=False,
    title="Edge scalar",
):
    """
    Color edges by a scalar attribute (e.g., Q).
    
    This function now calls plot_centerline_scalar_clean for improved visualization.
    """
    from .advanced_plots import plot_centerline_scalar_clean
    plot_centerline_scalar_clean(G, edge_attr=edge_attr, log_abs=log_abs, title=title)


def plot_flow_distribution(G, edge_attr="Q"):
    """
    Plot the distribution of flow values across edges.
    
    This function now calls plot_flow_distribution_clean for improved visualization.
    """
    from .advanced_plots import plot_flow_distribution_clean
    plot_flow_distribution_clean(G, edge_attr=edge_attr)


def plot_poiseuille_histograms(G, mu: float = 1.0e-3):
    """
    Compute radius, hydraulic resistance, mean velocity and wall shear
    directly from the graph and plot histograms (if data exists).
    """
    radii = []
    R_hyd = []
    u_mean = []
    tau_wall = []

    for u, v, data in G.edges(data=True):
        Q = data.get("Q", np.nan)
        if not np.isfinite(Q) or Q == 0.0:
            continue

        ru = G.nodes[u].get("radius", None)
        rv = G.nodes[v].get("radius", None)
        if ru is None and rv is None:
            continue
        elif ru is None:
            r = float(rv)
        elif rv is None:
            r = float(ru)
        else:
            r = 0.5 * (float(ru) + float(rv))
        if r <= 0.0:
            continue

        Pu = G.nodes[u].get("pressure", np.nan)
        Pv = G.nodes[v].get("pressure", np.nan)
        if not (np.isfinite(Pu) and np.isfinite(Pv)):
            continue

        length = data.get("length", None)
        if length is None:
            cu = np.asarray(G.nodes[u].get("coord", [0, 0, 0]), float)
            cv = np.asarray(G.nodes[v].get("coord", [0, 0, 0]), float)
            length = float(np.linalg.norm(cu - cv))
        length = max(float(length), 1e-8)

        dP = abs(Pu - Pv)
        R_edge = dP / abs(Q) if abs(Q) > 0 else np.nan
        A = pi * r**2
        u_edge = Q / A
        tau = 4.0 * mu * Q / (pi * r**3)

        radii.append(r)
        if np.isfinite(R_edge):
            R_hyd.append(R_edge)
        u_mean.append(u_edge)
        tau_wall.append(tau)

    def safe_hist(data, ax, title, xlabel):
        data = np.array(data, dtype=float)
        data = data[np.isfinite(data)]
        if data.size == 0:
            ax.text(0.5, 0.5, "No data", ha="center", va="center")
            ax.set_xticks([]); ax.set_yticks([])
            ax.set_title(title)
            return
        ax.hist(data, bins=30)
        ax.set_title(title)
        ax.set_xlabel(xlabel)

    fig, axes = plt.subplots(2, 2, figsize=(8, 6))
    safe_hist(radii, axes[0, 0], "Radius distribution", "radius (m)")
    safe_hist(R_hyd, axes[0, 1], "Hydraulic resistance", "R_hyd (Pa·s/m³)")
    safe_hist(u_mean, axes[1, 0], "Mean velocity", "u_mean (m/s)")
    safe_hist(tau_wall, axes[1, 1], "Wall shear stress", "τ_w (Pa)")
    plt.tight_layout()
    plt.show()


def summarize_poiseuille(G):
    """
    Print a basic summary of Poiseuille solution attached to G_centerline.
    """
    pressures = np.array([G.nodes[n].get("pressure", np.nan) for n in G.nodes], dtype=float)
    Q_edges = np.array([abs(data.get("Q", 0.0)) for _, _, data in G.edges(data=True)], dtype=float)
    tau = np.array([abs(data.get("tau_wall", 0.0)) for _, _, data in G.edges(data=True)], dtype=float)

    print("=== Poiseuille summary ===")
    print(f"Nodes           : {G.number_of_nodes()}")
    print(f"Edges           : {G.number_of_edges()}")
    print(f"Pressure min/max: {np.nanmin(pressures):.3e} / {np.nanmax(pressures):.3e} Pa")
    print(f"Flow |Q| min/max: {np.nanmin(Q_edges):.3e} / {np.nanmax(Q_edges):.3e} m^3/s")
    print(f"Wall shear |τ|  : {np.nanmin(tau):.3e} / {np.nanmax(tau):.3e} Pa")
