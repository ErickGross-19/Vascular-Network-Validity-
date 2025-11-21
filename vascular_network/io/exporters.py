import json
from pathlib import Path
from dataclasses import asdict
from typing import Any, Dict, Optional
import networkx as nx
import numpy as np

from ..models import ValidationReport


def centerline_graph_to_json(
    G: nx.Graph,
    node_pressures: Optional[Dict[Any, float]] = None,
    edge_flows: Optional[Dict[tuple, float]] = None,
) -> Dict[str, Any]:
    """
    Convert a centerline NetworkX graph (and optional Poiseuille results)
    into a JSON-serializable dict with 'nodes' and 'edges'.

    Each node includes:
      - 'id'
      - any existing node attributes (coord, radius, etc.)
      - optional 'pressure' from node_pressures

    Each edge includes:
      - 'u', 'v'
      - any existing edge attributes (length, etc.)
      - optional 'flow' from edge_flows
    """
    nodes_json: list[Dict[str, Any]] = []
    edges_json: list[Dict[str, Any]] = []

    node_pressures = node_pressures or {}
    edge_flows = edge_flows or {}

    for n, attrs in G.nodes(data=True):
        node_data: Dict[str, Any] = {
            "id": n if isinstance(n, (int, str)) else str(n)
        }

        for key, value in attrs.items():
            if isinstance(value, np.ndarray):
                node_data[key] = value.tolist()
            elif isinstance(value, (list, tuple)):
                node_data[key] = list(value)
            elif isinstance(value, (np.floating, np.integer)):
                node_data[key] = float(value)
            else:
                node_data[key] = value

        if n in node_pressures:
            node_data["pressure"] = float(node_pressures[n])

        nodes_json.append(node_data)

    for u, v, attrs in G.edges(data=True):
        edge_data: Dict[str, Any] = {
            "u": u if isinstance(u, (int, str)) else str(u),
            "v": v if isinstance(v, (int, str)) else str(v),
        }

        for key, value in attrs.items():
            if isinstance(value, np.ndarray):
                edge_data[key] = value.tolist()
            elif isinstance(value, (list, tuple)):
                edge_data[key] = list(value)
            elif isinstance(value, (np.floating, np.integer)):
                edge_data[key] = float(value)
            else:
                edge_data[key] = value

        if (u, v) in edge_flows:
            edge_data["flow"] = float(edge_flows[(u, v)])
        elif (v, u) in edge_flows:
            edge_data["flow"] = float(edge_flows[(v, u)])

        edges_json.append(edge_data)

    return {"nodes": nodes_json, "edges": edges_json}


def save_validation_report_json(
    report: ValidationReport,
    json_path: str | Path,
    centerline_graph: Optional[nx.Graph] = None,
    node_pressures: Optional[Dict[Any, float]] = None,
    edge_flows: Optional[Dict[tuple, float]] = None,
) -> None:
    """
    Save a ValidationReport to JSON, optionally including the full centerline graph.

    Parameters
    ----------
    report : ValidationReport
        The validation report to save.
    json_path : str or Path
        Path where the JSON file will be saved.
    centerline_graph : nx.Graph, optional
        If provided, the full centerline graph will be serialized and included.
    node_pressures : dict, optional
        Node pressures from Poiseuille analysis.
    edge_flows : dict, optional
        Edge flows from Poiseuille analysis.
    """
    json_path = Path(json_path)
    json_path.parent.mkdir(parents=True, exist_ok=True)

    data = asdict(report)

    if centerline_graph is not None:
        data["centerline_graph"] = centerline_graph_to_json(
            centerline_graph,
            node_pressures=node_pressures,
            edge_flows=edge_flows,
        )

    with open(json_path, "w") as f:
        json.dump(data, f, indent=2)

    print(f"[save_validation_report_json] Saved to {json_path}")
