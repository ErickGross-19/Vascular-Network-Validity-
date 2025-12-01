"""
Advanced visualization functions for centerline graphs and flow analysis.

These are the "_clean" versions from script (4) with improved plotting.
"""

import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from matplotlib.collections import LineCollection
from typing import Optional


def plot_centerline_graph_2d_edges(
    G: nx.Graph,
    plane: str = "xz",
    color_by: str = "radius",
    log_abs_Q: bool = True,
    title: str = "Centerline graph (2D, edge-based)",
):
    """
    Edge-based 2D plot of the centerline to avoid fake jumps.

    Parameters
    ----------
    G : networkx.Graph
        Centerline graph with node attribute 'coord' = (x,y,z) and
        optionally 'radius' (node) and 'Q' (edge flow).
    plane : {"xz", "xy", "yz"}
        Projection plane.
    color_by : {"radius", "Q", None}
        What to color edges by. If None, a single color is used.
    log_abs_Q : bool
        If True and color_by == "Q", use log10(|Q|) as color.
    """
    if plane == "xz":
        ix, iy = 0, 2
        xlabel, ylabel = "x", "z"
    elif plane == "xy":
        ix, iy = 0, 1
        xlabel, ylabel = "x", "y"
    elif plane == "yz":
        ix, iy = 1, 2
        xlabel, ylabel = "y", "z"
    else:
        raise ValueError("plane must be one of {'xz','xy','yz'}")

    segments = []
    colors = []

    for u, v, data in G.edges(data=True):
        cu = np.asarray(G.nodes[u].get("coord", (0, 0, 0)), dtype=float)
        cv = np.asarray(G.nodes[v].get("coord", (0, 0, 0)), dtype=float)

        x0, y0 = cu[ix], cu[iy]
        x1, y1 = cv[ix], cv[iy]
        segments.append(([x0, x1], [y0, y1]))

        if color_by == "radius":
            ru = float(G.nodes[u].get("radius", 0.0))
            rv = float(G.nodes[v].get("radius", 0.0))
            r_mean = 0.5 * (ru + rv)
            colors.append(r_mean)
        elif color_by == "Q":
            Q = float(data.get("Q", 0.0))
            if log_abs_Q:
                Q_abs = max(abs(Q), 1e-20)
                colors.append(np.log10(Q_abs))
            else:
                colors.append(Q)
        else:
            colors.append(1.0)

    plt.figure(figsize=(4, 6))
    if color_by is None:
        for (xs, ys) in segments:
            plt.plot(xs, ys, "-", linewidth=0.8)
    else:
        lines = [np.column_stack([xs, ys]) for xs, ys in segments]
        lc = LineCollection(lines, array=np.asarray(colors), cmap="viridis")
        lc.set_linewidth(1.0)
        fig, ax = plt.subplots(figsize=(4, 6))
        ax.add_collection(lc)
        ax.autoscale()
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        cbar = fig.colorbar(lc, ax=ax)
        cbar.set_label(color_by if color_by != "Q" else ("log10|Q|" if log_abs_Q else "|Q|"))
        plt.show()
        return

    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.gca().set_aspect("equal", adjustable="box")
    plt.show()


def plot_flow_distribution_clean(G: nx.Graph, edge_attr: str = "Q", qmin_abs: float = 1e-12):
    """
    Rank edges by |Q| and plot log10(|Q|) vs rank.
    Ignore edges below qmin_abs to avoid numerical noise.
    """
    qs = []
    for u, v, data in G.edges(data=True):
        if edge_attr in data:
            qs.append(abs(data[edge_attr]))
    qs = np.array(qs, dtype=float)

    qs = qs[qs > qmin_abs]
    if qs.size == 0:
        print("[plot_flow_distribution_clean] No edges above threshold.")
        return

    qs_sorted = np.sort(qs)[::-1]
    ranks = np.arange(1, len(qs_sorted) + 1)

    fig, ax = plt.subplots(figsize=(5, 3))
    ax.plot(ranks, np.log10(qs_sorted), ".", markersize=3)
    ax.set_xlabel("Edge rank (by |Q|)")
    ax.set_ylabel("log10(|Q|) [m³/s]")
    ax.set_title("Flow distribution across branches")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


def plot_centerline_scalar_clean(
    G: nx.Graph,
    edge_attr: str = "Q",
    log_abs: bool = True,
    title: Optional[str] = None,
):
    """
    Cleaner version of scalar plot along centerlines.
    """
    segments = []
    values = []

    for u, v, data in G.edges(data=True):
        c0 = np.asarray(G.nodes[u]["coord"], dtype=float)
        c1 = np.asarray(G.nodes[v]["coord"], dtype=float)
        segments.append(np.vstack([c0[[0, 2]], c1[[0, 2]]]))
        val = data.get(edge_attr, 0.0)
        if log_abs:
            val = np.log10(abs(val) + 1e-30)
        values.append(val)

    segments = np.array(segments)
    values = np.array(values)

    fig, ax = plt.subplots(figsize=(3, 6))

    lc = LineCollection(segments, cmap="viridis")
    lc.set_array(values)
    lc.set_linewidth(2.0)

    ax.add_collection(lc)
    ax.autoscale()
    ax.set_xlabel("x")
    ax.set_ylabel("z")
    if title is None:
        title = f"{edge_attr} along centerline"
        if log_abs:
            title += " (log10 |·|)"
    ax.set_title(title)
    cbar = fig.colorbar(lc, ax=ax)
    cbar.set_label(f"log10 |{edge_attr}|" if log_abs else edge_attr)
    plt.show()


def plot_flow_histograms(flow_metrics: dict, nbins: int = 60):
    """
    Plot histograms of flow metrics (Q, Re, u_mean, R_hyd).
    """
    Q = np.abs(flow_metrics["Q"])
    Re = flow_metrics["Re"]
    u = flow_metrics["u_mean"]
    Rh = flow_metrics["R_hyd"]

    fig, axes = plt.subplots(2, 2, figsize=(8, 6))

    ax = axes[0, 0]
    ax.hist(Q[Q > 0], bins=nbins)
    ax.set_xlabel("|Q|")
    ax.set_ylabel("count")
    ax.set_title("Flow magnitude |Q|")

    ax = axes[0, 1]
    ax.hist(np.log10(Q[Q > 0]), bins=nbins)
    ax.set_xlabel("log10 |Q|")
    ax.set_ylabel("count")
    ax.set_title("Flow distribution (log-scale)")

    ax = axes[1, 0]
    ax.hist(Re[Re > 0], bins=nbins)
    ax.set_xlabel("Re")
    ax.set_ylabel("count")
    ax.set_title("Reynolds number")

    ax = axes[1, 1]
    finite_Rh = Rh[np.isfinite(Rh)]
    if finite_Rh.size > 0:
        ax.hist(finite_Rh, bins=nbins)
    ax.set_xlabel("R_hyd (Pa·s/m^3)")
    ax.set_ylabel("count")
    ax.set_title("Hydraulic resistance")

    fig.tight_layout()
    plt.show()


def plot_length_and_radius_histograms(
    geom_metrics: dict,
    bins_length: int = 40,
    bins_radius: int = 40,
):
    """
    Plot histograms of edge lengths and radii from geometry metrics.
    """
    lengths = np.asarray(geom_metrics["edge_lengths"], dtype=float)
    radii = np.asarray(geom_metrics["edge_radii"], dtype=float)
    radii = radii[np.isfinite(radii)]

    fig, axes = plt.subplots(1, 2, figsize=(8, 3))

    ax = axes[0]
    ax.hist(lengths, bins=bins_length)
    ax.set_xlabel("Edge length")
    ax.set_ylabel("Count")
    ax.set_title("Centerline edge-length distribution")

    ax = axes[1]
    if radii.size > 0:
        ax.hist(radii, bins=bins_radius)
    ax.set_xlabel("Radius")
    ax.set_ylabel("Count")
    ax.set_title("Centerline radius distribution")

    fig.tight_layout()
    plt.show()


def plot_logQ_vs_radius(G: nx.Graph, min_Q: float = 1e-20):
    """
    Scatter plot of log10|Q| vs radius.

    For each edge, radius_mean = avg of node radii (or edge 'radius' if present).
    """
    radii = []
    logQ = []

    for u, v, data in G.edges(data=True):
        Q = float(data.get("Q", 0.0))
        if Q == 0.0:
            continue
        Q_abs = max(abs(Q), min_Q)
        logQ.append(np.log10(Q_abs))

        r_edge = data.get("radius", None)
        if r_edge is not None:
            r = float(r_edge)
        else:
            ru = float(G.nodes[u].get("radius", 0.0))
            rv = float(G.nodes[v].get("radius", 0.0))
            r = 0.5 * (ru + rv)
        radii.append(r)

    if not radii:
        print("[plot_logQ_vs_radius] No edges with nonzero Q found.")
        return

    radii = np.asarray(radii, dtype=float)
    logQ = np.asarray(logQ, dtype=float)

    plt.figure(figsize=(4, 4))
    plt.scatter(radii, logQ, s=5, alpha=0.5)
    plt.xlabel("Mean radius")
    plt.ylabel("log10 |Q|")
    plt.title("log10|Q| vs radius")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


def print_centerline_geometry_summary(geom_metrics: dict):
    """
    Print summary of centerline geometry metrics.
    """
    print("--- Centerline geometry ---")
    print(f"Nodes / edges     : {geom_metrics['n_nodes']} / {geom_metrics['n_edges']}")
    print(f"Leaves / branches : {geom_metrics['n_leaves']} / {geom_metrics['n_branch_nodes']}")
    print(f"Total length      : {geom_metrics['length_total']:.3f}")
    print(f"Mean / median L   : {geom_metrics['length_mean']:.3f} / {geom_metrics['length_median']:.3f}")
    print(f"Mean / median r   : {geom_metrics['radius_mean_global']:.4f} / {geom_metrics['radius_median_global']:.4f}")
    bb = geom_metrics["bounding_box"]
    print(f"Bounding box extents (dx, dy, dz): {bb['extents']}")


def print_flow_summary(flow_metrics: dict):
    """
    Print summary of flow metrics.
    """
    print("--- Flow metrics ---")
    print(f"Q_total           : {flow_metrics.get('Q_total', 'N/A'):.6g}")
    print(f"Q_mean / median   : {flow_metrics.get('Q_mean', 'N/A'):.6g} / {flow_metrics.get('Q_median', 'N/A'):.6g}")
    print(f"Re_max / median   : {flow_metrics.get('Re_max', 'N/A'):.3f} / {flow_metrics.get('Re_median', 'N/A'):.3f}")


def print_poiseuille_component_summary(comp_summaries: list):
    """
    Pretty-print per-component flow summaries.
    """
    print("\n=== Poiseuille components ===")
    for cs in comp_summaries:
        cid = cs["component_id"]
        print(f"\nComponent {cid}:")
        print(f"  nodes/edges  : {cs['num_nodes']} / {cs['num_edges']}")
        print(f"  z_min / z_max: {cs['z_min']:.3f} / {cs['z_max']:.3f}")
        if not cs["flow_solved"]:
            print(f"  FLOW NOT SOLVED: {cs.get('reason', 'unknown')}")
            continue
        print(f"  inlet node   : {cs['inlet']}")
        print(f"  outlet nodes : {cs['outlets']}")
        print(f"  Q_in / Q_out : {cs['total_inlet_flow']:.3e} / {cs['total_outlet_flow']:.3e}")
        if cs["warnings"]:
            print("  warnings     :")
            for w in cs["warnings"]:
                print("    -", w)
