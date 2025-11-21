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


def plot_centerline_graph_2d(G, plane="xz", title="Centerline graph"):
    """
    Plot a 2D projection of the centerline graph.
    """
    plt.figure(figsize=(4, 6))
    xs, ys = [], []

    for n, data in G.nodes(data=True):
        p = np.asarray(data.get("pos", data.get("coord", [0, 0, 0])), dtype=float)
        if plane == "xy":
            xs.append(p[0]); ys.append(p[1])
        elif plane == "yz":
            xs.append(p[1]); ys.append(p[2])
        else:
            xs.append(p[0]); ys.append(p[2])

    plt.scatter(xs, ys, s=5)
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
    """
    segments = []
    vals = []

    for u, v, data in G.edges(data=True):
        p0 = np.asarray(G.nodes[u].get("pos", G.nodes[u].get("coord", [0, 0, 0])), float)
        p1 = np.asarray(G.nodes[v].get("pos", G.nodes[v].get("coord", [0, 0, 0])), float)
        segments.append([p0[[0, 2]], p1[[0, 2]]])
        val = data.get(edge_attr, np.nan)
        vals.append(val)

    vals = np.array(vals, dtype=float)
    good = np.isfinite(vals)
    if not good.any():
        print(f"[plot_centerline_scalar] No finite values for '{edge_attr}'.")
        return

    vals = vals[good]
    segments = np.array(segments, dtype=float)[good]

    if log_abs:
        vals_plot = np.log10(np.abs(vals) + 1e-30)
    else:
        vals_plot = vals

    lc = LineCollection(segments, array=vals_plot, cmap=cmap, linewidth=2.0)

    fig, ax = plt.subplots(figsize=(4, 6))
    ax.add_collection(lc)
    ax.autoscale()
    ax.set_aspect("equal")
    ax.set_xlabel("x")
    ax.set_ylabel("z")
    ax.set_title(title)
    cbar = fig.colorbar(lc, ax=ax)
    cbar.set_label(f"log10(|{edge_attr}|)" if log_abs else edge_attr)
    plt.tight_layout()
    plt.show()


def plot_flow_distribution(G, edge_attr="Q"):
    """
    Plot the distribution of flow values across edges.
    """
    vals = []
    for _, _, data in G.edges(data=True):
        v = data.get(edge_attr, np.nan)
        if np.isfinite(v):
            vals.append(abs(v))
    vals = np.array(vals, dtype=float)
    if vals.size == 0:
        print("[plot_flow_distribution] No finite flow values.")
        return

    vals_sorted = np.sort(vals)[::-1]
    plt.figure(figsize=(4, 3))
    plt.plot(np.arange(len(vals_sorted)), vals_sorted, marker=".", linestyle="none")
    plt.xlabel("Edge rank (by |Q|)")
    plt.ylabel("|Q|")
    plt.title("Flow distribution across branches")
    plt.tight_layout()
    plt.show()


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
