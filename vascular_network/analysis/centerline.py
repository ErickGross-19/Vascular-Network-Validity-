import numpy as np
import networkx as nx
from scipy import ndimage
from skimage.morphology import skeletonize
from typing import Dict, Tuple, Any


def extract_centerline_graph(
    fluid_mask: np.ndarray,
    origin: np.ndarray,
    pitch: float,
):
    """
    Extract a centerline graph from a voxelized fluid domain.

    - Skeletonize the fluid (3D).
    - Distance transform -> local radius.
    - Build a graph with one node per skeleton voxel, radius, and coord.
    - Keep only the largest connected component.
    """
    skeleton = skeletonize(fluid_mask)
    dist_voxel = ndimage.distance_transform_edt(fluid_mask)
    dist_world = dist_voxel * pitch

    idx_i, idx_j, idx_k = np.where(skeleton)
    num_skel_voxels = len(idx_i)
    if num_skel_voxels == 0:
        raise RuntimeError("Skeletonization produced zero voxels.")

    G = nx.Graph()
    ijk_to_id: Dict[Tuple[int, int, int], int] = {}

    for node_id, (i, j, k) in enumerate(zip(idx_i, idx_j, idx_k)):
        ijk = (int(i), int(j), int(k))
        ijk_to_id[ijk] = node_id

        coord = origin + pitch * np.array([i, j, k], dtype=float)
        radius = float(dist_world[i, j, k])

        G.add_node(
            node_id,
            ijk=ijk,
            coord=coord,
            pos=coord,
            radius=radius,
        )

    neighbor_offsets = [
        (1, 0, 0), (-1, 0, 0),
        (0, 1, 0), (0, -1, 0),
        (0, 0, 1), (0, 0, -1),
    ]
    skel_set = set(ijk_to_id.keys())
    for ijk, node_id in ijk_to_id.items():
        i, j, k = ijk
        for di, dj, dk in neighbor_offsets:
            nb = (i + di, j + dj, k + dk)
            if nb in skel_set:
                nb_id = ijk_to_id[nb]
                if not G.has_edge(node_id, nb_id):
                    G.add_edge(node_id, nb_id)

    components = list(nx.connected_components(G))
    if not components:
        raise RuntimeError("Centerline graph ended up empty.")
    if len(components) > 1:
        largest = max(components, key=len)
        G = G.subgraph(largest).copy()

    for u, v in G.edges:
        cu = np.asarray(G.nodes[u]["coord"], dtype=float)
        cv = np.asarray(G.nodes[v]["coord"], dtype=float)
        length = float(np.linalg.norm(cu - cv))
        G.edges[u, v]["length"] = length

    centerline_meta = {
        "origin": origin.tolist(),
        "pitch": float(pitch),
        "num_nodes": G.number_of_nodes(),
        "num_edges": G.number_of_edges(),
    }

    return G, centerline_meta
