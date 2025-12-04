#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# Cell 1: imports & dataclasses

from __future__ import annotations

import json
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Optional, List, Dict, Any, Tuple

import numpy as np
import trimesh
from pymeshfix import MeshFix
import pyvista as pv
import networkx as nx

from scipy import ndimage
from skimage import measure
from skimage.morphology import skeletonize
import matplotlib.pyplot as plt
import runpy
import cadquery as cq
from math import pi
from skimage.measure import marching_cubes
import trimesh.smoothing as tmsmooth
import time
from contextlib import contextmanager
from scipy.ndimage import distance_transform_edt
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
import networkx as nx
from matplotlib.collections import LineCollection
from pathlib import Path




# Cell 1b: core dataclasses
@contextmanager
def timed_stage(name: str):
    """
    Context manager to print a simple progress message and timing
    for a pipeline stage.
    """
    print(f"[{name}] started...")
    t0 = time.time()
    try:
        yield
    finally:
        dt = time.time() - t0
        print(f"[{name}] finished in {dt:.2f} s")

@dataclass
class MeshDiagnostics:
    watertight: bool
    euler_number: int
    num_vertices: int
    num_faces: int
    num_components: int
    non_manifold_edges: int
    degenerate_faces: int
    bounding_box_extents: List[float]  # [dx, dy, dz]
    volume: Optional[float]            # in units^3, None if cannot compute
    volume_source: str                 # "mesh" | "convex_hull" | "unknown"



@dataclass
class SurfaceQuality:
    min_face_area: float
    max_face_area: float
    mean_face_area: float
    min_edge_length: float
    max_edge_length: float
    mean_edge_length: float
    max_aspect_ratio: float
    frac_aspect_ratio_over_10: float


@dataclass
class ValidationFlags:
    status: str               # "ok", "warnings", "fail"
    flags: List[str]


@dataclass
class ValidationReport:
    input_file: str
    intermediate_stl: Optional[str]
    cleaned_stl: str
    scafold_stl: str

    before: MeshDiagnostics
    after_basic_clean: MeshDiagnostics
    after_voxel: MeshDiagnostics
    after_repair: MeshDiagnostics

    flags: ValidationFlags

    surface_before: SurfaceQuality
    surface_after: SurfaceQuality

    connectivity: Dict[str, Any]
    centerline_summary: Dict[str, Any]
    poiseuille_summary: Dict[str, Any]


# In[ ]:


# Cell 2: basic mesh helpers
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

    # Serialize nodes
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

    # Serialize edges
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

        # Try (u, v) then (v, u) to find the flow
        if (u, v) in edge_flows:
            edge_data["flow"] = float(edge_flows[(u, v)])
        elif (v, u) in edge_flows:
            edge_data["flow"] = float(edge_flows[(v, u)])

        edges_json.append(edge_data)

    return {"nodes": nodes_json, "edges": edges_json}




def load_stl_mesh(path, process: bool = True) -> trimesh.Trimesh:
    """
    Load an STL into a single Trimesh.

    - Accepts a filename or Path.
    - Uses trimesh.load (lets trimesh handle opening and file_type detection).
    - If a Scene is returned, concatenates all geometries into one mesh.
    """
    path = Path(path)

    # Let trimesh handle file_type + opening
    mesh = trimesh.load(
        str(path),
        force="mesh",   # force mesh rather than scene when possible
        process=process,
    )

    # Some versions may still give a Scene → merge all geometries
    if isinstance(mesh, trimesh.Scene):
        if not mesh.geometry:
            raise ValueError(f"No geometry found in STL: {path}")
        mesh = trimesh.util.concatenate(mesh.dump())

    # Ensure it's really a Trimesh
    if not isinstance(mesh, trimesh.Trimesh):
        raise TypeError(f"Loaded object is not a Trimesh: {type(mesh)}")

    mesh.remove_unreferenced_vertices()
    return mesh

# Cell 2b: degenerate faces

def count_degenerate_faces(mesh: trimesh.Trimesh, area_eps: float = 1e-18) -> int:
    """
    Count faces whose area is effectively zero.
    """
    areas = mesh.area_faces
    return int(np.sum(areas <= area_eps))

def estimate_voxel_volume(
    mesh: trimesh.Trimesh,
    pitch: float = 0.05,
    method: str = "ray",
) -> float | None:
    """
    Estimate volume by voxelization:
      - voxelize mesh at given pitch
      - fill interior
      - count voxels * pitch^3

    Returns None if voxelization fails.
    """
    try:
        vox = mesh.voxelized(pitch, method=method)
        vox_filled = vox.fill()
        mask = vox_filled.matrix.astype(bool)
        num_voxels = int(mask.sum())
        return float(num_voxels * (pitch ** 3))
    except Exception:
        return None

# Cell 2c: diagnostics

def compute_diagnostics(mesh: trimesh.Trimesh, volume_pitch: float = 0.05,
    max_voxels: float = 3e7,) -> MeshDiagnostics:
    """
    Compute core mesh health/topology metrics, including approximate volume.
    """
    watertight = mesh.is_watertight
    euler = mesh.euler_number

    num_vertices = int(mesh.vertices.shape[0])
    num_faces = int(mesh.faces.shape[0])

    components = mesh.split(only_watertight=False)
    num_components = len(components)

    # Non-manifold edges: edges shared by more than 2 faces
    try:
        if mesh.edges_unique is not None and mesh.edges_unique_inverse is not None:
            counts = np.bincount(mesh.edges_unique_inverse)
            non_manifold_edges = int(np.sum(counts > 2))
        else:
            non_manifold_edges = 0
    except AttributeError:
        non_manifold_edges = 0

    degenerate_faces = count_degenerate_faces(mesh)

    extents = mesh.extents.tolist()

    # Volume: if watertight use mesh.volume, else convex hull
       # Volume + source
    volume = estimate_voxel_volume(mesh, pitch=volume_pitch)
    volume_source = "voxel_estimate"

    return MeshDiagnostics(
        watertight=watertight,
        euler_number=int(euler),
        num_vertices=num_vertices,
        num_faces=num_faces,
        num_components=num_components,
        non_manifold_edges=non_manifold_edges,
        degenerate_faces=degenerate_faces,
        bounding_box_extents=extents,
        volume=volume,
        volume_source=volume_source,
    )

# Cell 2d: surface quality (Part A)

def compute_surface_quality(mesh: trimesh.Trimesh) -> SurfaceQuality:
    """
    Compute basic surface quality metrics:
      - face areas
      - edge lengths
      - triangle aspect ratios
    """
    v = mesh.vertices
    f = mesh.faces

    # Face areas
    areas = mesh.area_faces
    min_face_area = float(areas.min())
    max_face_area = float(areas.max())
    mean_face_area = float(areas.mean())

    # Edge lengths (unique edges)
    edges = mesh.edges_unique
    lengths = np.linalg.norm(v[edges[:, 0]] - v[edges[:, 1]], axis=1)
    min_edge_length = float(lengths.min())
    max_edge_length = float(lengths.max())
    mean_edge_length = float(lengths.mean())

    # Triangle aspect ratio: longest edge / shortest altitude
    v0 = v[f[:, 0]]
    v1 = v[f[:, 1]]
    v2 = v[f[:, 2]]

    e01 = np.linalg.norm(v1 - v0, axis=1)
    e12 = np.linalg.norm(v2 - v1, axis=1)
    e20 = np.linalg.norm(v0 - v2, axis=1)

    longest_edge = np.maximum(e01, np.maximum(e12, e20))
    h = 2.0 * areas / (longest_edge + 1e-16)
    aspect_ratio = longest_edge / (h + 1e-16)

    max_aspect_ratio = float(aspect_ratio.max())
    frac_over_10 = float(np.mean(aspect_ratio > 10.0))

    return SurfaceQuality(
        min_face_area=min_face_area,
        max_face_area=max_face_area,
        mean_face_area=mean_face_area,
        min_edge_length=min_edge_length,
        max_edge_length=max_edge_length,
        mean_edge_length=mean_edge_length,
        max_aspect_ratio=max_aspect_ratio,
        frac_aspect_ratio_over_10=frac_over_10,
    )


# In[ ]:


# Cell 3: basic clean
def basic_clean(
    mesh: trimesh.Trimesh,
    min_component_faces: int = 50,
    fill_holes: bool = True,
    max_hole_area: float | None = None,
) -> trimesh.Trimesh:
    """
    Lightweight but robust cleanup focused on:
      - keeping only the main component
      - removing junk faces
      - optional small-hole filling
      - fixing normals

    This leans more towards producing a single, clean lumen
    than preserving tiny details or exact volume.
    """
    mesh = mesh.copy()

    # 1) Drop tiny disconnected components (e.g. floating triangles)
    components = mesh.split(only_watertight=False)
    if len(components) > 1:
        kept = [c for c in components if c.faces.shape[0] >= min_component_faces]
        if not kept:
            # If everything is tiny, keep the largest only
            kept = [max(components, key=lambda c: c.faces.shape[0])]
        mesh = trimesh.util.concatenate(kept)

    mesh.remove_unreferenced_vertices()

    # 2) Remove duplicate faces
    unique_faces = mesh.unique_faces()
    mesh.update_faces(unique_faces)
    mesh.remove_unreferenced_vertices()

    # 3) Remove degenerate faces (near-zero area)
    areas = mesh.area_faces
    keep = areas > 1e-18
    if keep.sum() == 0:
        # if we killed everything, just keep original faces
        keep = np.ones_like(areas, dtype=bool)
    mesh.update_faces(keep)
    mesh.remove_unreferenced_vertices()

    # 4) Optionally fill small holes
    if fill_holes:
        try:
            # max_hole_area=None -> fill all holes
            trimesh.repair.fill_holes(mesh, max_hole_area=max_hole_area)
        except Exception:
            pass

    # 5) Merge strictly identical vertices
    try:
        mesh.merge_vertices()
    except Exception:
        pass

    # 6) Fix normals & winding (if graph engine available)
    try:
        trimesh.repair.fix_normals(mesh, multibody=True)
        trimesh.repair.fix_winding(mesh)
    except Exception:
        pass

    mesh.remove_unreferenced_vertices()
    return mesh

# -------------------------------
# Cell 3b: voxel remesh + smooth (connectivity + smoothing priority)
# -------------------------------

def auto_adjust_voxel_pitch(
    mesh: trimesh.Trimesh,
    requested_pitch: float,
    max_voxels: float = 3e7,
    min_pitch: float = 0.02,
) -> float:
    """
    Given a mesh and requested voxel pitch, compute a safe pitch
    such that (dx*dy*dz) / pitch^3 <= max_voxels.

    Returns the adjusted pitch (>= requested_pitch, >= min_pitch).
    """
    extents = mesh.bounding_box.extents.astype(float)
    dx, dy, dz = extents
    volume_box = dx * dy * dz

    if max_voxels <= 0 or volume_box <= 0:
        return max(requested_pitch, min_pitch)

    pitch_min = (volume_box / max_voxels) ** (1.0 / 3.0)

    # Only relax upward (coarser) to keep grid size reasonable
    adjusted = max(requested_pitch, pitch_min, min_pitch)
    return float(adjusted)


def voxel_remesh_and_smooth(
    mesh: trimesh.Trimesh,
    pitch: float = 0.1,
    smooth_iters: int = 40,
    use_taubin: bool = True,
    dilation_iters: int = 1,
    closing_iters: int = 1,
    opening_iters: int = 1,
    max_voxels: float = 3e7,
) -> trimesh.Trimesh:
    """
    Voxel remesh + morphology + smoothing.

    Design goals:
      - produce a SINGLE, well-connected lumen
      - aggressively smooth away jaggies & tiny artifacts
      - do NOT try to preserve exact volume

    Steps:
      1) Auto-adjust pitch to keep voxel grid size sane.
      2) Voxelize to binary mask, relax pitch if trimesh complains.
      3) Keep only largest voxel connected component.
      4) Closing (fills small gaps) and opening (removes spikes).
      5) Dilation (thicken thin branches a bit).
      6) Marching cubes back to surface.
      7) Taubin or Laplacian smoothing.
    """
    mesh = mesh.copy()

    # 1) Auto-adjust pitch against bounding box
    pitch_eff = auto_adjust_voxel_pitch(
        mesh, requested_pitch=pitch, max_voxels=max_voxels
    )

    # 2) Voxelization with robustness to 'max_iter exceeded'
    for attempt in range(4):
        try:
            vox = mesh.voxelized(pitch_eff)
            break
        except ValueError as e:
            print(
                f"[voxel_remesh_and_smooth] voxelized(pitch={pitch_eff:.4g}) failed ({e}), "
                f"increasing pitch..."
            )
            pitch_eff *= 1.5
    else:
        raise RuntimeError(
            f"Voxelization failed even after relaxing pitch; final pitch={pitch_eff:.4g}"
        )

    vox_filled = vox.fill()
    fluid_mask = vox_filled.matrix.astype(bool)

    if not fluid_mask.any():
        raise RuntimeError(
            "Voxelization produced an empty fluid mask. Check pitch/scale."
        )

    # 3) Keep only the largest connected component in voxel space
    structure = ndimage.generate_binary_structure(3, 1)  # 6-connectivity
    labels, num_labels = ndimage.label(fluid_mask, structure=structure)
    if num_labels > 1:
        sizes = ndimage.sum(fluid_mask, labels, index=range(1, num_labels + 1))
        largest_label = 1 + int(np.argmax(sizes))
        fluid_mask = labels == largest_label

    # 4) Morphological closing/opening
    if closing_iters > 0:
        fluid_mask = ndimage.binary_closing(fluid_mask, iterations=closing_iters)

    if opening_iters > 0:
        fluid_mask = ndimage.binary_opening(fluid_mask, iterations=opening_iters)

    # 5) Dilation (helps weld thin broken parts)
    if dilation_iters > 0:
        fluid_mask = ndimage.binary_dilation(fluid_mask, iterations=dilation_iters)

    # 6) Marching cubes
    volume_uint8 = fluid_mask.astype(np.uint8)
    if volume_uint8.max() == 0:
        raise RuntimeError(
            "All voxels were removed after morphology; nothing left to remesh."
        )

    verts, faces, _, _ = marching_cubes(
        volume=volume_uint8,
        level=0.5,
        spacing=(pitch_eff, pitch_eff, pitch_eff),
    )

    mesh_voxel = trimesh.Trimesh(
        vertices=verts,
        faces=faces.astype(np.int64),
        process=False,
    )
    mesh_voxel.remove_unreferenced_vertices()

    # 7) Smoothing
    if smooth_iters > 0:
        try:
            if use_taubin:
                tmsmooth.filter_taubin(
                    mesh_voxel,
                    lamb=0.5,
                    nu=-0.53,
                    iterations=smooth_iters,
                )
            else:
                tmsmooth.filter_laplacian(
                    mesh_voxel,
                    lamb=0.5,
                    iterations=smooth_iters,
                )
        except Exception as e:
            print(f"[voxel_remesh_and_smooth] smoothing failed: {e}")

    mesh_voxel.remove_unreferenced_vertices()
    return mesh_voxel


def match_volume(
    mesh: trimesh.Trimesh,
    target_volume: float,
    max_scale_factor: float = 3.0,
) -> trimesh.Trimesh:
    """
    Uniformly scale mesh so its volume approx equals target_volume.

    - Computes scale = (target / current)^(1/3).
    - Clamps scale to [1/max_scale_factor, max_scale_factor] to avoid
      insane jumps (e.g., if current volume is tiny or huge).
    """
    mesh = mesh.copy()

    try:
        vol_now = float(mesh.volume)
    except Exception:
        vol_now = 0.0

    if vol_now <= 0 or target_volume <= 0:
        # nothing sensible to do
        return mesh

    raw_scale = (target_volume / vol_now) ** (1.0 / 3.0)

    # clamp
    min_s = 1.0 / max_scale_factor
    max_s = max_scale_factor
    scale = float(np.clip(raw_scale, min_s, max_s))

    center = mesh.center_mass
    mesh.vertices -= center
    mesh.vertices *= scale
    mesh.vertices += center

    mesh.remove_unreferenced_vertices()
    return mesh



# -------------------------------
# MeshFix repair (focus on main shell)
# -------------------------------

def meshfix_repair(
    mesh: trimesh.Trimesh,
    keep_largest_component: bool = True,
) -> trimesh.Trimesh:
    """
    Run MeshFix to enforce watertightness and fix topological issues,
    then optionally keep only the largest connected component.

    This version favors a single, clean lumen over preserving
    extra tiny islands or exact original volume.
    """
    mesh = mesh.copy()

    v = np.asarray(mesh.vertices, dtype=float)
    f = np.asarray(mesh.faces, dtype=np.int64)

    fixer = MeshFix(v, f)
    fixer.repair(
        verbose=False,
        joincomp=True,              # try to join components
        remove_smallest_components=False,
    )

    # Get repaired vertices/faces from MeshFix (version-safe)
    if hasattr(fixer, "points"):
        v_repaired = np.asarray(fixer.points, dtype=float)
    else:
        v_repaired = np.asarray(fixer.v, dtype=float)

    if hasattr(fixer, "faces"):
        f_repaired = np.asarray(fixer.faces, dtype=np.int64)
    else:
        f_repaired = np.asarray(fixer.f, dtype=np.int64)

    repaired = trimesh.Trimesh(
        vertices=v_repaired,
        faces=f_repaired,
        process=False,
    )
    repaired.remove_unreferenced_vertices()

    # Optionally keep only the largest connected component
    if keep_largest_component:
        comps = repaired.split(only_watertight=False)
        if len(comps) > 1:
            largest = max(comps, key=lambda c: c.faces.shape[0])
            repaired = largest.copy()
            repaired.remove_unreferenced_vertices()

    # Final normals/winding clean-up
    try:
        trimesh.repair.fix_normals(repaired, multibody=True)
        trimesh.repair.fix_winding(repaired)
    except Exception:
        pass

    repaired.remove_unreferenced_vertices()
    return repaired


# In[ ]:


# Cell 4: fluid mask from mesh
def mesh_to_fluid_mask(mesh, pitch):
    vox = mesh.voxelized(pitch)
    vox_filled = vox.fill()

    # Dense boolean occupancy
    fluid_mask = vox_filled.matrix.astype(bool)

    # Origin: use attribute if present, otherwise infer from transform
    if hasattr(vox_filled, "origin"):
        origin = np.array(vox_filled.origin, dtype=float)
    else:
        # translation part of the 4x4 transform
        origin = np.array(vox_filled.transform[:3, 3], dtype=float)

    # Pitch: handle both scalar and vector cases
    pitch_attr = getattr(vox_filled, "pitch", pitch)
    pitch_arr = np.asarray(pitch_attr, dtype=float)

    if pitch_arr.size == 1:
        pitch_val = float(pitch_arr)
    else:
        # assume isotropic voxels, use the first component (or np.mean(pitch_arr))
        pitch_val = float(pitch_arr[0])

    return fluid_mask, origin, pitch_val


# Cell 4b: connectivity analysis (Part B)

def analyze_connectivity_voxel(
    mesh: trimesh.Trimesh,
    pitch: float = 0.1,
) -> Tuple[Dict[str, Any], np.ndarray, np.ndarray, float]:
    """
    Analyze connectivity in voxel space:
      - number of fluid components
      - components that touch the bounding box ("ports")
      - trapped fluid components and volume fraction

    Returns
    -------
    connectivity_info : dict
    fluid_mask        : (nx, ny, nz) bool
    origin            : (3,) float
    pitch             : float
    """
    fluid_mask, origin, pitch = mesh_to_fluid_mask(mesh, pitch=pitch)
    nx, ny, nz = fluid_mask.shape
    num_fluid_voxels = int(fluid_mask.sum())
    if num_fluid_voxels == 0:
        raise RuntimeError("Fluid mask has no voxels; check voxelization or mesh.")

    # Connected components (6-connectivity)
    structure = ndimage.generate_binary_structure(rank=3, connectivity=1)
    labels, num_labels = ndimage.label(fluid_mask, structure=structure)

    component_sizes = ndimage.sum(fluid_mask, labels, index=range(1, num_labels + 1))
    component_sizes = [int(s) for s in component_sizes]

    # Port voxels: fluid on any boundary face of the voxel domain
    port_mask = np.zeros_like(fluid_mask, dtype=bool)
    port_mask[0, :, :] |= fluid_mask[0, :, :]
    port_mask[-1, :, :] |= fluid_mask[-1, :, :]
    port_mask[:, 0, :] |= fluid_mask[:, 0, :]
    port_mask[:, -1, :] |= fluid_mask[:, -1, :]
    port_mask[:, :, 0] |= fluid_mask[:, :, 0]
    port_mask[:, :, -1] |= fluid_mask[:, :, -1]

    port_labels = np.unique(labels[port_mask & (labels > 0)])
    port_labels = [int(l) for l in port_labels if l != 0]

    if len(port_labels) > 0:
        reachable_mask = np.isin(labels, port_labels)
        reachable_voxels = int(reachable_mask.sum())
        reachable_fraction = reachable_voxels / num_fluid_voxels
    else:
        reachable_voxels = 0
        reachable_fraction = 0.0

    all_labels = np.arange(1, num_labels + 1)
    trapped_labels = sorted(list(set(all_labels) - set(port_labels)))
    trapped_sizes = [component_sizes[l - 1] for l in trapped_labels]

    connectivity_info = {
        "pitch": pitch,
        "shape": (nx, ny, nz),
        "num_fluid_voxels": num_fluid_voxels,
        "num_fluid_components": int(num_labels),
        "component_sizes": component_sizes,
        "num_port_components": len(port_labels),
        "port_component_labels": port_labels,
        "reachable_fraction": float(reachable_fraction),
        "num_trapped_components": len(trapped_labels),
        "trapped_component_labels": [int(l) for l in trapped_labels],
        "trapped_component_sizes": trapped_sizes,
    }

    return connectivity_info, fluid_mask, origin, pitch

# Cell 4c: centerline graph extraction (Part C)

# --- Cell: centerline extraction (updated) ---

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
    # 1) Skeleton
    skeleton = skeletonize(fluid_mask)
    dist_voxel = ndimage.distance_transform_edt(fluid_mask)
    dist_world = dist_voxel * pitch

    idx_i, idx_j, idx_k = np.where(skeleton)
    num_skel_voxels = len(idx_i)
    if num_skel_voxels == 0:
        raise RuntimeError("Skeletonization produced zero voxels.")

    G = nx.Graph()
    ijk_to_id: Dict[Tuple[int, int, int], int] = {}

    # 2) Nodes
    for node_id, (i, j, k) in enumerate(zip(idx_i, idx_j, idx_k)):
        ijk = (int(i), int(j), int(k))
        ijk_to_id[ijk] = node_id

        coord = origin + pitch * np.array([i, j, k], dtype=float)
        radius = float(dist_world[i, j, k])

        # Store both coord and pos (for plotting)
        G.add_node(
            node_id,
            ijk=ijk,
            coord=coord,
            pos=coord,   # <-- for plotting functions that expect 'pos'
            radius=radius,
        )

    # 3) Edges (6-connectivity)
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

    # 4) Keep only largest connected component
    components = list(nx.connected_components(G))
    if not components:
        raise RuntimeError("Centerline graph ended up empty.")
    if len(components) > 1:
        # You can print if you want
        # print(f"[extract_centerline_graph] {len(components)} components; keeping largest.")
        largest = max(components, key=len)
        G = G.subgraph(largest).copy()

    # 5) Edge lengths
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

# --- Cell: Poiseuille network (updated) ---

def compute_poiseuille_network(
    G: nx.Graph,
    mu: float = 1.0,
    inlet_nodes=None,
    outlet_nodes=None,
    pin: float = 1.0,
    pout: float = 0.0,
    write_to_graph: bool = True,
):
    """
    Solve a Poiseuille flow network on a centerline graph.

    Auto-BC mode:
      - If inlet_nodes and outlet_nodes are both None:
          * find all degree-1 nodes (leaves)
          * choose inlet = leaf with minimum z
          * choose outlet = leaf with maximum z
    """

    warnings_list: list[str] = []

    # 0) Components (for summary only now, we already made G one component upstream)
    components = list(nx.connected_components(G))

    # 1) Edge conductances ------------------------------------------------
    edge_conductance: dict[tuple[int, int], float] = {}

    def edge_G(u, v) -> float:
        data = G.edges[u, v]

        # Radius: prefer edge radius, else average node radii
        ru = G.nodes[u].get("radius", None)
        rv = G.nodes[v].get("radius", None)
        r_edge = data.get("radius", None)

        if r_edge is not None:
            r = float(r_edge)
        elif ru is None and rv is None:
            return 0.0
        elif ru is None:
            r = float(rv)
        elif rv is None:
            r = float(ru)
        else:
            r = 0.5 * (float(ru) + float(rv))

        if r <= 0.0:
            return 0.0

        length = data.get("length", None)
        if length is None:
            cu = np.asarray(G.nodes[u].get("coord", [0, 0, 0]), dtype=float)
            cv = np.asarray(G.nodes[v].get("coord", [0, 0, 0]), dtype=float)
            length = float(np.linalg.norm(cu - cv))
        length = max(float(length), 1e-8)

        return (pi * r**4) / (8.0 * mu * length)

    for u, v in G.edges:
        G_uv = edge_G(u, v)
        edge_conductance[(u, v)] = G_uv
        edge_conductance[(v, u)] = G_uv

    # 2) Boundary nodes ---------------------------------------------------
    leaves = [n for n in G.nodes if G.degree(n) == 1]

    if inlet_nodes is None and outlet_nodes is None:
        if len(leaves) < 2:
            warnings_list.append(
                "Less than two leaf nodes; cannot auto-select inlet/outlet."
            )
            inlet_nodes = []
            outlet_nodes = []
        else:
            # use Z coordinate of node "coord"
            leaves_with_z = []
            for n in leaves:
                coord = np.asarray(G.nodes[n].get("coord", [0, 0, 0]), dtype=float)
                z = coord[2] if coord.size >= 3 else 0.0
                leaves_with_z.append((n, z))

            leaves_with_z.sort(key=lambda x: x[1])  # sort by z
            inlet_nodes = [leaves_with_z[0][0]]      # min z
            outlet_nodes = [leaves_with_z[-1][0]]    # max z
    else:
        inlet_nodes = list(inlet_nodes) if inlet_nodes is not None else []
        outlet_nodes = list(outlet_nodes) if outlet_nodes is not None else []

    inlet_nodes = [n for n in inlet_nodes if n in G]
    outlet_nodes = [n for n in outlet_nodes if n in G]

    if not inlet_nodes or not outlet_nodes:
        warnings_list.append(
            "Missing inlets or outlets after validation; system may be singular."
        )

    boundary_nodes = set(inlet_nodes) | set(outlet_nodes)
    interior_nodes = [n for n in G.nodes if n not in boundary_nodes]

    boundary_pressure = {n: float(pin) for n in inlet_nodes}
    for n in outlet_nodes:
        boundary_pressure.setdefault(n, float(pout))

    # 3) Linear system ----------------------------------------------------
    n_int = len(interior_nodes)
    used_lstsq = False

    if n_int > 0:
        idx = {n: i for i, n in enumerate(interior_nodes)}
        A = np.zeros((n_int, n_int), dtype=float)
        b = np.zeros(n_int, dtype=float)

        for n in interior_nodes:
            i = idx[n]
            for m in G.neighbors(n):
                G_nm = edge_conductance.get((n, m), 0.0)
                if G_nm == 0.0:
                    continue
                if m in interior_nodes:
                    j = idx[m]
                    A[i, i] += G_nm
                    A[i, j] -= G_nm
                else:
                    Pm = boundary_pressure.get(m, 0.0)
                    A[i, i] += G_nm
                    b[i] += G_nm * Pm

        rank_A = int(np.linalg.matrix_rank(A))
        if rank_A < A.shape[0]:
            warnings_list.append("Matrix is rank-deficient; using least-squares.")

        try:
            P_int = np.linalg.solve(A, b)
        except np.linalg.LinAlgError:
            P_int, *_ = np.linalg.lstsq(A, b, rcond=None)
            used_lstsq = True

        node_pressures: dict[int, float] = {}
        for n in interior_nodes:
            node_pressures[n] = float(P_int[idx[n]])
        for n, P in boundary_pressure.items():
            node_pressures[n] = float(P)

        matrix_shape = A.shape
        matrix_rank = rank_A
    else:
        # degenerate case: everything is boundary
        node_pressures = {
            n: float(boundary_pressure.get(n, pout)) for n in G.nodes
        }
        matrix_shape = (0, 0)
        matrix_rank = 0

    # 4) Edge flows + totals ---------------------------------------------
    edge_flows: dict[tuple[int, int], float] = {}
    for u, v in G.edges:
        G_uv = edge_conductance.get((u, v), 0.0)
        if G_uv == 0.0:
            edge_flows[(u, v)] = 0.0
            continue
        Pu = node_pressures[u]
        Pv = node_pressures[v]
        edge_flows[(u, v)] = float(G_uv * (Pu - Pv))

    total_inlet_flow = 0.0
    total_outlet_flow = 0.0

    for n in inlet_nodes:
        flux = 0.0
        for m in G.neighbors(n):
            G_nm = edge_conductance.get((n, m), 0.0)
            if G_nm == 0.0:
                continue
            flux += G_nm * (node_pressures[n] - node_pressures[m])
        total_inlet_flow += flux

    for n in outlet_nodes:
        flux = 0.0
        for m in G.neighbors(n):
            G_nm = edge_conductance.get((n, m), 0.0)
            if G_nm == 0.0:
                continue
            flux += G_nm * (node_pressures[n] - node_pressures[m])
        total_outlet_flow += -flux

    summary = {
        "num_nodes": int(G.number_of_nodes()),
        "num_edges": int(G.number_of_edges()),
        "num_components": int(len(components)),
        "num_inlets": int(len(inlet_nodes)),
        "num_outlets": int(len(outlet_nodes)),
        "num_interior": int(len(interior_nodes)),
        "matrix_shape": tuple(matrix_shape),
        "matrix_rank": int(matrix_rank),
        "used_lstsq": bool(used_lstsq),
        "total_inlet_flow": float(total_inlet_flow),
        "total_outlet_flow": float(total_outlet_flow),
        "total_inflow": float(total_inlet_flow),
        "total_outflow": float(total_outlet_flow),
        "pin": float(pin),
        "pout": float(pout),
        "warnings": warnings_list,
        "inlet_node_ids": [str(n) for n in inlet_nodes],
        "outlet_node_ids": [str(n) for n in outlet_nodes],
    }

    # 5) Write back to graph ---------------------------------------------
    if write_to_graph:
        for n, P in node_pressures.items():
            G.nodes[n]["pressure"] = P
        for (u, v), Q in edge_flows.items():
            if G.has_edge(u, v):
                G.edges[u, v]["Q"] = Q

    return {
        "node_pressures": node_pressures,
        "edge_flows": edge_flows,
        "summary": summary,
        "G": G,
    }


def ensure_poiseuille_solved(G: nx.Graph, delta_p: float = 1.0, mu: float = 1.0e-3):
    """
    If G has no 'pressure' yet, pick one inlet and one outlet (min-z, max-z
    leaf) and run Poiseuille; otherwise return G unchanged.
    """
    if any("pressure" in data for _, data in G.nodes(data=True)):
        return G

    res = compute_poiseuille_network(
        G,
        mu=mu,
        inlet_nodes=None,
        outlet_nodes=None,
        pin=delta_p,
        pout=0.0,
        write_to_graph=True,
    )
    return res["G"]



def make_scaffold_shell_from_fluid(
    fluid_mask: np.ndarray,
    pitch: float,
    wall_thickness: float = 0.4,
):
    """
    Build a scaffold as a narrow shell around the fluid domain.

    Parameters
    ----------
    fluid_mask : (nx, ny, nz) bool
        True for fluid voxels (inside channels), False elsewhere.
    pitch : float
        Voxel size in world units (mm, etc.).
    wall_thickness : float
        Desired wall thickness of the scaffold around channels.

    Returns
    -------
    scaffold_mesh : trimesh.Trimesh
        Mesh of the scaffold shell (no big filled-in regions).
    """
    # 1) Distance from each solid voxel to nearest fluid voxel
    # fluid_mask==True inside lumen; we want distance OUTSIDE the lumen
    dist_vox = distance_transform_edt(~fluid_mask)  # distance in voxels
    dist = dist_vox * pitch                         # distance in world units

    # 2) Make a shell: outside fluid, within wall_thickness of channels
    shell_mask = (~fluid_mask) & (dist <= wall_thickness)

    if not shell_mask.any():
        raise RuntimeError("Shell mask is empty; try increasing wall_thickness.")

    # 3) Marching cubes on shell_mask
    vol_uint8 = shell_mask.astype(np.uint8)
    verts, faces, _, _ = marching_cubes(
        volume=vol_uint8,
        level=0.5,
        spacing=(pitch, pitch, pitch),
    )

    scaffold_mesh = trimesh.Trimesh(
        vertices=verts,
        faces=faces.astype(np.int64),
        process=False,
    )
    scaffold_mesh.remove_unreferenced_vertices()
    return scaffold_mesh



# In[ ]:


# Cell 5: CAD Python (.py) loader & STL export (CadQuery)

def cad_python_to_stl(
    cad_py_path: str | Path,
    stl_out_path: str | Path | None = None,
    tolerance: float = 0.1,
) -> Path:
    """
    Execute a CadQuery Python script and export the resulting solid to STL.

    Assumptions about the CAD Python file:
    - It uses CadQuery (import cadquery as cq).
    - It defines either:
      - a function `build()` that returns a CadQuery Workplane / Shape, OR
      - a variable `model`, `result`, or `assembly` holding the Workplane / Shape.

    Parameters
    ----------
    cad_py_path : str or Path
        Path to the CAD Python script.
    stl_out_path : str or Path or None
        Where to save the exported STL. If None, use <stem>_raw.stl in the same folder.
    tolerance : float
        Tessellation tolerance; adjust based on units & detail.

    Returns
    -------
    Path
        Path to the generated STL file.
    """
    if cq is None:
        raise ImportError(
            "cadquery is not available. Install it to process CAD Python files."
        )

    cad_py_path = Path(cad_py_path)
    if stl_out_path is None:
        stl_out_path = cad_py_path.with_name(cad_py_path.stem + "_raw.stl")
    stl_out_path = Path(stl_out_path)

    # Execute the CAD Python script in an isolated namespace
    namespace = runpy.run_path(str(cad_py_path))

    solid = None

    # Preferred: a build() function that returns the final model
    if "build" in namespace and callable(namespace["build"]):
        solid = namespace["build"]()
    else:
        # Try a few conventional variable names
        for name in ["model", "result", "assembly", "wm", "wp"]:
            if name in namespace:
                solid = namespace[name]
                break

    if solid is None:
        raise ValueError(
            "Could not find a CadQuery object. "
            "Define a build() function or a 'model'/'result' variable in the CAD script."
        )

    # Export the solid to STL
    # Note: exporters can handle Workplane, Shape, Assembly, etc.
    cq.exporters.export(solid, str(stl_out_path), tolerance=tolerance)

    return stl_out_path


# In[ ]:


# Cell 6: main pipeline + report + JSON export
def save_validation_report_json(
    report: ValidationReport,
    json_path: str | Path,
    G_centerline: Optional[nx.Graph] = None,
    network_results: Optional[Dict[str, Any]] = None,
):
    """
    Convert ValidationReport (and optionally the full centerline/Poiseuille network)
    to a JSON-friendly dict and write it to disk.
    """
    json_path = Path(json_path)

    # Optional network pieces
    centerline_graph_json = None
    poiseuille_network_json = None

    if G_centerline is not None:
        node_pressures = None
        edge_flows = None
        if network_results is not None:
            node_pressures = network_results.get("node_pressures")
            edge_flows = network_results.get("edge_flows")

        centerline_graph_json = centerline_graph_to_json(
            G_centerline,
            node_pressures=node_pressures,
            edge_flows=edge_flows,
        )

    if network_results is not None:
        poiseuille_network_json = {
            "node_pressures": {
                str(k): float(v) for k, v in network_results.get("node_pressures", {}).items()
            },
            "edge_flows": {
                f"{u}->{v}": float(q) for (u, v), q in network_results.get("edge_flows", {}).items()
            },
        }

    data: Dict[str, Any] = {
        "input_file": report.input_file,
        "intermediate_stl": report.intermediate_stl,
        "cleaned_stl": report.cleaned_stl,
        "before": asdict(report.before),
        "after_basic_clean": asdict(report.after_basic_clean),
        "after_voxel": asdict(report.after_voxel),
        "after_repair": asdict(report.after_repair),
        "flags": {
            "status": report.flags.status,
            "flags": report.flags.flags,
        },
        "surface_before": asdict(report.surface_before),
        "surface_after": asdict(report.surface_after),
        "connectivity": report.connectivity,
        "centerline": report.centerline_summary,
        "poiseuille": report.poiseuille_summary,
    }

    # Attach network content if available
    if centerline_graph_json is not None:
        data["centerline_graph"] = centerline_graph_json

    if poiseuille_network_json is not None:
        data["poiseuille_network"] = poiseuille_network_json

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def validate_and_repair_geometry(
    input_path: str | Path,
    cleaned_stl_path: Optional[str | Path] = None,
    scaffold_stl_path: Optional[str | Path] = None,   # <--- new
    wall_thickness: float = 0.4, 
    report_path: Optional[str | Path] = None,
    cad_tessellation_tolerance: float = 0.1,
    voxel_pitch: float = 0.1,
    smooth_iters: int = 40,
    dilation_iters: int = 0,
    inlet_nodes: Optional[Sequence[Any]] = None,
    outlet_nodes: Optional[Sequence[Any]] = None,
):
    """
    Full pipeline:

    - Load STL or CadQuery .py
    - Basic clean
    - Voxel remesh + smooth
    - MeshFix repair
    - Diagnostics + surface quality (before/after)
    - Connectivity (B)
    - Centerlines (C)
    - Poiseuille network (D)
    - Save cleaned STL and JSON report (optional)
    """
    input_path = Path(input_path)

    # Decide how to get to the first STL
    intermediate_stl_path: Optional[Path]
    if input_path.suffix.lower() == ".stl":
        intermediate_stl_path = input_path
    elif input_path.suffix.lower() == ".py":
        intermediate_stl_path = input_path.with_suffix("_tess.stl")
        cad_python_to_stl(input_path, intermediate_stl_path, tolerance=cad_tessellation_tolerance)
    else:
        raise ValueError(f"Unsupported input type: {input_path.suffix}")

    # Where to write the final cleaned STL
    if cleaned_stl_path is None:
        cleaned_stl_path = intermediate_stl_path.with_name(
            intermediate_stl_path.stem + "_cleaned.stl"
        )
    cleaned_stl_path = Path(cleaned_stl_path)

    # 1) Load original STL
    mesh_original = load_stl_mesh(intermediate_stl_path)
    diag_before = compute_diagnostics(mesh_original)
    surf_before = compute_surface_quality(mesh_original)

    # 2) Basic clean
    mesh_clean = basic_clean(mesh_original)
    diag_after_basic = compute_diagnostics(mesh_clean)

    # 3) Voxel remesh + smooth
    # Only use volume correction if the original mesh volume is reliable.
    mesh_voxel = voxel_remesh_and_smooth(
        mesh_clean,
        pitch=voxel_pitch,          # or 0.03–0.1 depending on resolution
        smooth_iters=smooth_iters,
        dilation_iters=2,
        closing_iters=1,
        opening_iters=1,
    )

    diag_after_voxel = compute_diagnostics(mesh_voxel)


    # 4) MeshFix repair on voxel mesh
    mesh_repaired = meshfix_repair(mesh_voxel)

    diag_after_repair = compute_diagnostics(mesh_repaired)
    surf_after = compute_surface_quality(mesh_repaired)

    # 5) Flags & status
    flags_list: List[str] = []
    neg_flags: List[str] = []

    if not diag_before.watertight and diag_after_repair.watertight:
        flags_list.append("fixed_watertightness")
    if diag_before.num_components != diag_after_repair.num_components:
        flags_list.append("changed_component_count")

    if not diag_after_repair.watertight:
        neg_flags.append("not_watertight")
    if diag_after_repair.num_components > 1:
        neg_flags.append("multiple_components")
    if diag_after_repair.non_manifold_edges > 0:
        neg_flags.append("non_manifold_edges")
    if diag_after_repair.degenerate_faces > 0:
        neg_flags.append("degenerate_faces")

    if any(f in ("not_watertight", "multiple_components") for f in neg_flags):
        status = "fail"
    elif neg_flags:
        status = "warnings"
    else:
        status = "ok"

    flags = ValidationFlags(
        status=status,
        flags=flags_list + neg_flags,
    )

    # 6) Connectivity (B)
    connectivity_info, fluid_mask, origin, pitch = analyze_connectivity_voxel(
        mesh_repaired,
        pitch=voxel_pitch,
    )

    # 7) Centerlines (C)
    G_centerline, centerline_meta = extract_centerline_graph(
        fluid_mask,
        origin=origin,
        pitch=pitch,
    )

    wall_thickness = 0.4   # e.g. 0.4 mm; tune to your print/material constraints
    scaffold_mesh = make_scaffold_shell_from_fluid(fluid_mask, pitch, wall_thickness)

    if scaffold_stl_path is None:
        intermediate_stl_path = Path(intermediate_stl_path)
        scaffold_stl_path = intermediate_stl_path.with_name(
            intermediate_stl_path.stem + "_scaffold_shell.stl"
        )
    else:
        scaffold_stl_path = Path(scaffold_stl_path)

    scaffold_mesh.export(scaffold_stl_path)
    print("Scaffold STL saved at:", scaffold_stl_path)



    # Summarize centerline radii for JSON
    radii = [float(data["radius"]) for _, data in G_centerline.nodes(data=True)]
    centerline_summary = {
        "meta": centerline_meta,
        "radius_min": float(min(radii)) if radii else 0.0,
        "radius_max": float(max(radii)) if radii else 0.0,
        "radius_mean": float(np.mean(radii)) if radii else 0.0,
    }

    # 8) Poiseuille network (D)
    network_results = compute_poiseuille_network(
        G_centerline,
        mu=1.0,
        inlet_nodes=inlet_nodes,
        outlet_nodes=outlet_nodes,
        pin=1.0,
        pout=0.0,
    )

    poiseuille_summary = network_results["summary"]

    # 9) Export cleaned STL
    mesh_repaired.export(cleaned_stl_path)

    # 10) Build ValidationReport object
    report = ValidationReport(
        input_file=str(input_path),
        intermediate_stl=str(intermediate_stl_path),
        cleaned_stl=str(cleaned_stl_path),
        scafold_stl=str(scaffold_stl_path),
        before=diag_before,
        after_basic_clean=diag_after_basic,
        after_voxel=diag_after_voxel,
        after_repair=diag_after_repair,
        flags=flags,
        surface_before=surf_before,
        surface_after=surf_after,
        connectivity=connectivity_info,
        centerline_summary=centerline_summary,
        poiseuille_summary=poiseuille_summary,
    )


    # 11) Optionally dump JSON
    if report_path is not None:
        save_validation_report_json(
            report,
            report_path,
            G_centerline=G_centerline,
            network_results=network_results,
        )

    return report, G_centerline


# In[ ]:


# Cell 7:  Core helpers 

def load_report_json(json_path: str | Path) -> Dict[str, Any]:
    json_path = Path(json_path)
    with open(json_path, "r", encoding="utf-8") as f:
        return json.load(f)


def pretty_print_report_json(json_path: str | Path) -> None:
    """
    Load a saved JSON report and pretty-print it to stdout.
    """
    data = load_report_json(json_path)
    print(json.dumps(data, indent=2, sort_keys=True))


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
    # support both naming schemes
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


# ---------- Mesh surface-quality plots ----------


def plot_surface_quality(mesh: trimesh.Trimesh, title: str) -> None:
    """
    Simple surface-quality visualization:
      - histogram of face areas
      - histogram of edge lengths
    """
    mesh = mesh.copy()

    # Face areas
    areas = mesh.area_faces

    # Edge lengths
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
    # For plotting we typically don't want trimesh to auto-fix things,
    # so use process=False.
    mesh_orig = load_stl_mesh(report.input_file, process=False)
    mesh_clean = load_stl_mesh(report.cleaned_stl, process=False)

    plot_surface_quality(mesh_orig, "Original")
    plot_surface_quality(mesh_clean, "Cleaned")



# ---------- Diagnostics plots from ValidationReport ----------

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

    # Bar for reachable fraction
    if reachable_fraction is not None:
        plt.figure(figsize=(4, 4))
        plt.bar(["reachable_fraction"], [reachable_fraction])
        plt.ylim(0, 1.0)
        plt.title("Connectivity reachable_fraction")
        plt.tight_layout()

    # Bar for component count if present
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


# ---------- Network / centerline visualization ----------

# --- Cell: plotting helpers (updated) ---

def plot_centerline_graph_2d(G, plane="xz", title="Centerline graph"):
    plt.figure(figsize=(4, 6))
    xs, ys = [], []

    for n, data in G.nodes(data=True):
        p = np.asarray(data.get("pos", data.get("coord", [0, 0, 0])), dtype=float)
        if plane == "xy":
            xs.append(p[0]); ys.append(p[1])
        elif plane == "yz":
            xs.append(p[1]); ys.append(p[2])
        else:  # "xz"
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
        segments.append([p0[[0, 2]], p1[[0, 2]]])  # XZ projection
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

# ---------- E. Small text summary ----------
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


# ---------- High-level convenience wrapper ----------

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

    # Pipeline diagnostics
    plot_volume_pipeline(report)
    plot_components_pipeline(report)

    # Surface quality
    plot_surface_quality_from_report(report)

    # Connectivity
    plot_connectivity_summary(report)

    # Centerline radii summary
    plot_centerline_radii_summary(report)

    # Poiseuille flows
    plot_poiseuille_flows_from_report(report)

    # Centerline graph projection (if provided)
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




# In[ ]:


# Example for an STL input
report, G_centerline = validate_and_repair_geometry("C:/Users/Erick/Downloads/Vascular_Network_Original.stl", voxel_pitch=0.05, smooth_iters=30, dilation_iters=1)
#report, G_centerline = validate_and_repair_geometry(""C:/Users/Erick/Downloads/liver_vasculature_schematic_cylinder_branches.stl", voxel_pitch=0.05, smooth_iters=20)

# Example for a CadQuery Python file input
#report = validate_and_repair_geometry("C:/Users/Erick/Downloads/Test-Branching_Strucutre.py", voxel_pitch=0.01, smooth_iters=20)

show_full_report(report, G_centerline)


# In[ ]:


def show_original_and_cleaned(original_stl: str, cleaned_stl: str):
    """Visualize original and cleaned mesh side-by-side in a trimesh Scene."""
    m_orig = load_stl_mesh(original_stl)
    m_clean = load_stl_mesh(cleaned_stl)

    # Shift cleaned mesh a bit in +X so they don't overlap
    m_clean_shifted = m_clean.copy()
    offset = m_orig.extents[0] * 1.2  # 20% gap
    m_clean_shifted.apply_translation([offset, 0.0, 0.0])

    # Color them differently
    m_orig.visual.face_colors = [200, 50, 50, 255]    # reddish
    m_clean_shifted.visual.face_colors = [50, 200, 50, 255]  # greenish

    scene = trimesh.Scene()
    scene.add_geometry(m_orig)
    scene.add_geometry(m_clean_shifted)

    return scene.show()  # opens viewer (Pyglet or WebGL, depending on environment)

orig = "C:/Users/Erick/Downloads/connected_vascular_network_1_2_4_8_4_2_1.stl"
clean = "C:/Users/Erick/Downloads/connected_vascular_network_1_2_4_8_4_2_1_cleaned.stl"

show_original_and_cleaned(orig, clean)


# In[ ]:


# Cell 2: Interactive inlet/outlet selection + Poiseuille solver

import numpy as np
import networkx as nx
import ipywidgets as widgets
from IPython.display import display
from scipy.sparse import lil_matrix
from scipy.sparse.linalg import spsolve


# ---------- 1. Build hydraulic resistances ----------
def assign_hydraulic_resistance(G, mu=1.0e-3):
    """
    Ensure each edge has 'R_hyd' based on length & radius_mean:
        R = 8 * mu * L / (pi * R^4)
    mu in Pa·s, L and R in consistent length units.
    """
    for u, v, data in G.edges(data=True):
        L = float(data.get("length", 0.0))
        r = float(data.get("radius_mean", 0.0))
        if L <= 0 or r <= 0:
            R_hyd = np.inf
        else:
            R_hyd = 8.0 * mu * L / (np.pi * r**4)
        data["R_hyd"] = R_hyd


# ---------- 2. Solve Poiseuille network (single inlet/outlet, fixed Δp) ----------
def solve_poiseuille_network(
    G,
    inlet_node,
    outlet_node,
    delta_p,
    mu=1.0e-3,
):
    """
    Solve Poiseuille flow on a centerline graph with one inlet & one outlet.
    Assumes:
      - edges have 'length' and 'radius_mean' (for R_hyd),
      - graph is connected.

    Boundary conditions:
      p[inlet_node]  = +delta_p
      p[outlet_node] = 0

    Returns
    -------
    results : dict
        Contains:
        - 'pressures': dict node -> p
        - 'Q_edges'  : dict (u,v) -> Q
        - 'Q_in'     : total flow at inlet (m^3/s)
        - 'R_eff'    : effective network resistance Δp / Q_in
        - 'tau_max'  : max wall shear across edges
    """
    G = G.copy()

    # ensure R_hyd present
    assign_hydraulic_resistance(G, mu=mu)

    nodes = list(G.nodes)
    n = len(nodes)
    idx = {node: i for i, node in enumerate(nodes)}

    inlet = inlet_node
    outlet = outlet_node

    fixed_p = {
        inlet: float(delta_p),
        outlet: 0.0,
    }

    # identify unknown-pressure nodes
    unknown_nodes = [n for n in nodes if n not in fixed_p]
    m = len(unknown_nodes)

    A = lil_matrix((m, m), dtype=float)
    b = np.zeros(m, dtype=float)

    # assemble conservation equations at unknown nodes
    for row_i, node in enumerate(unknown_nodes):
        p_i_row = row_i
        for nbr in G.neighbors(node):
            data = G.edges[node, nbr]
            R_e = float(data.get("R_hyd", np.inf))
            if not np.isfinite(R_e) or R_e <= 0:
                continue

            conductance = 1.0 / R_e

            if nbr in fixed_p:
                # neighbor has fixed pressure
                b[p_i_row] += conductance * fixed_p[nbr]
                A[p_i_row, p_i_row] += conductance
            else:
                # neighbor unknown pressure
                j = unknown_nodes.index(nbr)
                A[p_i_row, p_i_row] += conductance
                A[p_i_row, j] -= conductance

    # solve A p_unknown = b
    if m > 0:
        p_unknown = spsolve(A.tocsr(), b)
    else:
        p_unknown = np.array([], dtype=float)

    # assemble full pressure vector
    pressures = {}
    for node in nodes:
        if node in fixed_p:
            pressures[node] = fixed_p[node]
        else:
            k = unknown_nodes.index(node)
            pressures[node] = float(p_unknown[k])

    # attach to graph
    nx.set_node_attributes(G, values=pressures, name="pressure")

    # compute flows & wall shear on edges
    Q_edges = {}
    tau_max = 0.0

    for u, v, data in G.edges(data=True):
        R_e = float(data.get("R_hyd", np.inf))
        if not np.isfinite(R_e) or R_e <= 0:
            Q = 0.0
        else:
            dp = pressures[u] - pressures[v]
            Q = dp / R_e

        data["Q"] = Q

        L = float(data.get("length", 0.0))
        r = float(data.get("radius_mean", 0.0))
        if r > 0:
            A_xsec = np.pi * r**2
            u_mean = Q / A_xsec
            tau_wall = 4.0 * mu * u_mean / r   # τ = 4 μ U / R
        else:
            u_mean = 0.0
            tau_wall = 0.0

        data["u_mean"] = u_mean
        data["tau_wall"] = tau_wall

        tau_max = max(tau_max, abs(tau_wall))
        Q_edges[(u, v)] = Q

    # total flow at inlet (sum of outgoing flows)
    Q_in = 0.0
    for nbr in G.neighbors(inlet):
        data = G.edges[inlet, nbr]
        Q_in += data["Q"]  # sign already correct via dp/R

    # effective network resistance
    R_eff = delta_p / Q_in if Q_in != 0 else np.inf

    results = {
        "G": G,  # solved graph
        "pressures": pressures,
        "Q_edges": Q_edges,
        "Q_in": Q_in,
        "R_eff": R_eff,
        "tau_max": tau_max,
    }
    return results


# ---------- 3. Find candidate inlet/outlet nodes ----------
def find_boundary_nodes(G):
    """
    Return a list of degree-1 nodes (likely inlets/outlets).
    """
    return [n for n in G.nodes if G.degree[n] == 1]


# ---------- 4. Interactive widget ----------
def make_poiseuille_widget(G_centerline, mu=1.0e-3, default_delta_p=1000.0):
    """
    Build an ipywidgets UI for selecting inlet/outlet and Δp,
    solving Poiseuille flow, and plotting the result.
    """
    boundary_nodes = find_boundary_nodes(G_centerline)
    if not boundary_nodes:
        boundary_nodes = list(G_centerline.nodes)  # fallback

    inlet_dropdown = widgets.Dropdown(
        options=boundary_nodes,
        description="Inlet:",
        layout=widgets.Layout(width="250px"),
    )

    outlet_dropdown = widgets.Dropdown(
        options=boundary_nodes,
        description="Outlet:",
        layout=widgets.Layout(width="250px"),
    )

    delta_p_slider = widgets.FloatSlider(
        value=default_delta_p,
        min=10.0,
        max=5000.0,
        step=10.0,
        description="Δp (Pa):",
        continuous_update=False,
        readout_format=".0f",
        layout=widgets.Layout(width="400px"),
    )

    run_button = widgets.Button(
        description="Solve Poiseuille",
        button_style="success",
        layout=widgets.Layout(width="200px"),
    )

    output_area = widgets.Output()

    def on_run_clicked(b):
        with output_area:
            output_area.clear_output()

            inlet = inlet_dropdown.value
            outlet = outlet_dropdown.value
            delta_p = float(delta_p_slider.value)

            if inlet == outlet:
                print("Inlet and outlet must be different.")
                return

            # Solve
            res = solve_poiseuille_network(
                G_centerline,
                inlet_node=inlet,
                outlet_node=outlet,
                delta_p=delta_p,
                mu=mu,
            )
            G_sol = res["G"]

            # Summary
            print(f"Inlet node : {inlet}")
            print(f"Outlet node: {outlet}")
            print(f"Δp         : {delta_p:.3e} Pa")
            print(f"Q_in       : {res['Q_in']:.3e} m³/s")
            print(f"R_eff      : {res['R_eff']:.3e} Pa·s/m³")
            print(f"τ_wall max : {res['tau_max']:.3e} Pa")
            print()

            # Plots
            summarize_poiseuille(G_sol)
            plot_centerline_scalar(G_sol, edge_attr="Q", log_abs=True,
                                   title="Centerlines colored by log10 |Q|")
            plot_flow_distribution(G_sol, edge_attr="Q")
            plot_poiseuille_histograms(G_sol)

    run_button.on_click(on_run_clicked)

    ui = widgets.VBox(
        [
            inlet_dropdown,
            outlet_dropdown,
            delta_p_slider,
            run_button,
        ]
    )
    display(ui, output_area)


# To launch the interactive panel once G_centerline is available:
make_poiseuille_widget(G_centerline, mu=1.0e-3, default_delta_p=1000.0)


# In[ ]:


import numpy as np
from pathlib import Path
import trimesh
from skimage.measure import marching_cubes


def inlay_channels_into_box_voxel(
    channel_stl_path: str | Path,
    box_size: tuple[float, float, float],
    box_center: tuple[float, float, float] | None = None,
    pitch: float = 0.15,
    output_stl_path: str | Path | None = None,
) -> tuple[Path, trimesh.Trimesh, trimesh.Trimesh]:
    """
    Create a solid box with the channel STL as negative space (voids),
    using a voxel-based boolean (box - channels).

    This version avoids MemoryError by doing mesh.contains() slice-by-slice
    in the z-direction instead of for the full 3D grid at once.

    Parameters
    ----------
    channel_stl_path : str or Path
        Path to the channel (fluid-domain) STL. Should be cleaned & watertight.
    box_size : (Lx, Ly, Lz)
        Box dimensions in same units as the STL (e.g., mm).
    box_center : (cx, cy, cz) or None
        Center of the box. If None, center on the channel mesh bounding box.
    pitch : float
        Voxel size in same units as the STL (mm). Smaller -> more detail, more memory.
    output_stl_path : str or Path or None
        Where to save the scaffold STL. If None, use <stem>_box_scaffold_voxel.stl.

    Returns
    -------
    output_stl_path : Path
        Path to the scaffold STL file.
    channel_mesh : trimesh.Trimesh
        The loaded channel mesh.
    scaffold_mesh : trimesh.Trimesh
        The resulting solid scaffold mesh (box minus channels).
    """
    channel_stl_path = Path(channel_stl_path)

    # 1. Load the channel mesh
    channel_mesh = load_stl_mesh(channel_stl_path)

    # 2. Box placement
    bounds = channel_mesh.bounds  # [[minx, miny, minz], [maxx, maxy, maxz]]
    channel_center = bounds.mean(axis=0)
    if box_center is None:
        box_center = tuple(channel_center.tolist())
    box_center = np.array(box_center, dtype=float)

    Lx, Ly, Lz = box_size
    half = 0.5 * np.array(box_size, dtype=float)

    box_min = box_center - half
    box_max = box_center + half

    # 3. Build voxel grid coordinates
    xs = np.arange(box_min[0], box_max[0] + pitch, pitch)
    ys = np.arange(box_min[1], box_max[1] + pitch, pitch)
    zs = np.arange(box_min[2], box_max[2] + pitch, pitch)

    nx, ny, nz = len(xs), len(ys), len(zs)
    if nx <= 1 or ny <= 1 or nz <= 1:
        raise ValueError("Box size is too small relative to pitch to create a voxel grid.")

    # 4. Allocate scaffold occupancy:
    # 1 = solid box, 0 = void (channel)
    scaffold_mask = np.ones((nx, ny, nz), dtype=bool)

    # Precompute XY grid once for speed
    XX, YY = np.meshgrid(xs, ys, indexing="ij")  # shape (nx, ny)

    # 5. Determine which voxel centers are inside the channel, slice-by-slice
    # This is the memory-safe part: we never build the full 3D grid at once.
    for kz, z in enumerate(zs):
        # points in this z-slice: (nx*ny, 3)
        Z = np.full_like(XX, z, dtype=float)
        pts_slice = np.column_stack([XX.ravel(), YY.ravel(), Z.ravel()])

        inside_ch_slice = channel_mesh.contains(pts_slice)
        inside_ch_slice = inside_ch_slice.reshape(nx, ny)

        # scaffold is "box minus channels": set to False where inside channel
        scaffold_mask[:, :, kz] = ~inside_ch_slice

    # Check we have some solid voxels left
    if not scaffold_mask.any():
        raise RuntimeError("Voxel boolean (box - channels) produced no scaffold voxels (all void).")

    # 6. Marching cubes on scaffold volume
    volume_uint8 = scaffold_mask.astype(np.uint8)
    verts, faces, _, _ = marching_cubes(
        volume=volume_uint8,
        level=0.5,
        spacing=(pitch, pitch, pitch),
    )

    # 7. Shift verts from voxel index space to world coords (add box_min)
    verts_world = verts + box_min

    scaffold_mesh = trimesh.Trimesh(
        vertices=verts_world,
        faces=faces.astype(np.int64),
        process=False,
    )
    scaffold_mesh.remove_unreferenced_vertices()

    # 8. Save to STL
    if output_stl_path is None:
        output_stl_path = channel_stl_path.with_name(
            channel_stl_path.stem + "_box_scaffold_voxel.stl"
        )
    else:
        output_stl_path = Path(output_stl_path)

    scaffold_mesh.export(output_stl_path)

    return output_stl_path, channel_mesh, scaffold_mesh


# In[ ]:


cleaned = "C:/Users/Erick/Downloads/connected_vascular_network_1_2_4_8_4_2_1_cleaned.stl"

# Use after-repair extents and add a margin
dx, dy, dz = report.after_repair.bounding_box_extents
box_size = (dx + 2.0, dy + 2.0, dz + 2.0)  # add ~1 mm margin on each side

scaffold_path, ch_mesh, scaffold_mesh = inlay_channels_into_box_voxel(
    cleaned,
    box_size=box_size,
    box_center=None,   # center box on channel mesh
    pitch=0.15,        # adjust to 0.1 if you want finer detail
)

print("Scaffold STL saved at:", scaffold_path)

diag_channels = compute_diagnostics(ch_mesh)
diag_scaffold = compute_diagnostics(scaffold_mesh)

print("Channel extents :", diag_channels.bounding_box_extents)
print("Channel volume  :", diag_channels.volume)
print("Scaffold extents:", diag_scaffold.bounding_box_extents)
print("Scaffold volume :", diag_scaffold.volume)


# View scaffold alone
scene = trimesh.Scene()
scene.add_geometry(scaffold_mesh)
scene.show()


# In[ ]:


ch_colored = ch_mesh.copy()
ch_colored.visual.face_colors = [50, 200, 50, 150]   # semi-transparent green

scaffold_colored = scaffold_mesh.copy()
scaffold_colored.visual.face_colors = [200, 50, 50, 255]  # solid red

scene = trimesh.Scene()
scene.add_geometry(scaffold_colored)
scene.add_geometry(ch_colored)
scene.show()


# In[ ]:





# In[ ]:


def test_straight_cylinder(
    stl_path: str | Path,
    radius: float,
    length: float,
    voxel_pitches=(0.2, 0.1, 0.05, 0.025),
    smooth_iters: int = 10,
    dilation_iters: int = 1,
    mu: float = 1.0,
    pin: float = 1.0,
    pout: float = 0.0,
):
    stl_path = Path(stl_path)

    vols_before = []
    vols_after = []
    flow_numeric = []
    flow_error_rel = []

    print(f"=== Straight cylinder test: {stl_path} ===")
    print(f"R = {radius}, L = {length}\n")

    # Analytic Poiseuille flow
    delta_p = pin - pout
    Q_ref = math.pi * radius**4 * delta_p / (8.0 * mu)
    print(f"Analytic Poiseuille Q_ref = {Q_ref:.6g}\n")

    for pitch in voxel_pitches:
        print(f"--- voxel_pitch = {pitch} ---")

        report, G = validate_and_repair_geometry(
            stl_path,
            voxel_pitch=pitch,
            smooth_iters=smooth_iters,
            dilation_iters=dilation_iters,
            inlet_nodes=None,   # auto
            outlet_nodes=None,  # auto
        )

        V_before = report.before.volume
        V_after  = report.after_repair.volume
        vols_before.append(V_before)
        vols_after.append(V_after)

        conn = report.connectivity
        n_comp = conn.get("num_fluid_components", None)
        reachable = conn.get("reachable_fraction", None)

        print(f"V_before = {V_before:.6g}, V_after = {V_after:.6g}")
        print(f"num_fluid_components = {n_comp}, reachable_fraction = {reachable:.3f}")

        # Use Poiseuille summary from report
        poi = report.poiseuille_summary
        Q_num = poi.get("total_inlet_flow", poi.get("total_inflow", None))
        flow_numeric.append(Q_num)

        if Q_num is not None:
            rel_err = abs(Q_num - Q_ref) / abs(Q_ref)
            flow_error_rel.append(rel_err)
            print(f"Q_numeric = {Q_num:.6g}, rel_error = {rel_err:.3e}\n")
        else:
            flow_error_rel.append(np.nan)
            print("Q_numeric missing in summary.\n")

    # === Plots ===
    voxel_pitches = np.asarray(voxel_pitches, dtype=float)
    vols_before = np.asarray(vols_before, dtype=float)
    vols_after = np.asarray(vols_after, dtype=float)
    flow_numeric = np.asarray(flow_numeric, dtype=float)
    flow_error_rel = np.asarray(flow_error_rel, dtype=float)

    # Volume vs voxel_pitch
    plt.figure()
    plt.plot(voxel_pitches, vols_before, marker="o", label="V_before")
    plt.plot(voxel_pitches, vols_after, marker="s", label="V_after")
    plt.xlabel("voxel_pitch")
    plt.ylabel("Volume")
    plt.title("Cylinder: volume vs voxel_pitch")
    plt.legend()
    plt.gca().invert_xaxis()  # often nicer: finer pitch on the right

    # Flow-relative-error vs voxel_pitch
    plt.figure()
    plt.plot(voxel_pitches, flow_error_rel, marker="o")
    plt.xlabel("voxel_pitch")
    plt.ylabel("Relative error |Q_num - Q_ref| / Q_ref")
    plt.title("Cylinder: flow error vs voxel_pitch")
    plt.gca().invert_xaxis()
    plt.yscale("log")  # usually a log scale is more informative

    plt.show()

test_straight_cylinder(
    "C:/Users/Erick/Downloads/generated_stls/straight_cylinder.stl",
    radius=0.5,
    length=5.0,
    voxel_pitches=(0.2, 0.1, 0.05, 0.025),
)


# In[ ]:


def test_y_branch(
    stl_path: str | Path,
    R_in: float,
    L_in: float,
    R_out1: float,
    L_out1: float,
    R_out2: float,
    L_out2: float,
    voxel_pitch: float = 0.05,
    smooth_iters: int = 10,
    dilation_iters: int = 1,
    mu: float = 1.0,
    pin: float = 1.0,
    pout: float = 0.0,
):
    stl_path = Path(stl_path)
    print(f"=== Y-branch test: {stl_path} ===\n")

    report, G = validate_and_repair_geometry(
        stl_path,
        voxel_pitch=voxel_pitch,
        smooth_iters=smooth_iters,
        dilation_iters=dilation_iters,
        inlet_nodes=None,
        outlet_nodes=None,
    )

    # Identify leaf nodes and classify inlet/outlets by coordinates
    leaves = [n for n in G.nodes if G.degree(n) == 1]
    coords = {n: np.asarray(G.nodes[n].get("coord", [0, 0, 0]), dtype=float) for n in leaves}

    # inlet: leaf with min z
    inlet = min(leaves, key=lambda n: coords[n][2])

    # outlets: the other leaves, choose two with max z, then split by x sign
    leaves_not_inlet = [n for n in leaves if n != inlet]
    # take top 2 by z
    leaves_sorted_by_z = sorted(leaves_not_inlet, key=lambda n: coords[n][2], reverse=True)
    cand1, cand2 = leaves_sorted_by_z[:2]

    # assign by x < 0 / x > 0
    if coords[cand1][0] < coords[cand2][0]:
        outlet1, outlet2 = cand1, cand2
    else:
        outlet1, outlet2 = cand2, cand1

    print(f"Inlet node: {inlet}, coord = {coords[inlet]}")
    print(f"Outlet1 node: {outlet1}, coord = {coords[outlet1]}")
    print(f"Outlet2 node: {outlet2}, coord = {coords[outlet2]}\n")

    # Re-run Poiseuille with explicit boundaries so we know exactly who's what
    results = compute_poiseuille_network(
        G,
        mu=mu,
        inlet_nodes=[inlet],
        outlet_nodes=[outlet1, outlet2],
        pin=pin,
        pout=pout,
    )

    node_pressures = results["node_pressures"]
    edge_flows = results["edge_flows"]

    # Compute net flow at each outlet from edge_flows
    def net_outflow(node):
        q = 0.0
        for (u, v), Q_uv in edge_flows.items():
            if u == node:
                q += Q_uv      # flow leaving node
            elif v == node:
                q -= Q_uv      # flow entering node
        return q

    Q1 = net_outflow(outlet1)
    Q2 = net_outflow(outlet2)

    print(f"Q1 (outlet1) = {Q1:.6g}")
    print(f"Q2 (outlet2) = {Q2:.6g}")

    # Analytic ratio ~ (R1^4 / L1) / (R2^4 / L2)
    ratio_ref = (R_out1**4 / L_out1) / (R_out2**4 / L_out2)
    ratio_num = abs(Q1) / abs(Q2) if (Q1 != 0 and Q2 != 0) else np.nan

    print(f"Analytic ratio Q1/Q2 ≈ {ratio_ref:.6g}")
    print(f"Numeric ratio Q1/Q2  = {ratio_num:.6g}")
    print(f"Relative error       = {abs(ratio_num - ratio_ref) / abs(ratio_ref):.3e}")


test_y_branch(
    "C:/Users/Erick/Downloads/generated_stls/y_branch.stl",
    R_in=0.5,   L_in=5.0,
    R_out1=0.4, L_out1=4.0,
    R_out2=0.2, L_out2=4.0,
)


# In[ ]:


def demo_skeleton_failure(stl_path: str | Path, voxel_pitch: float = 1.0):

    stl_path = Path(stl_path)
    print(f"=== Skeletonization failure demo: {stl_path} ===")

    # Load & basic clean
    mesh = load_stl_mesh(stl_path)
    diag = compute_diagnostics(mesh)
    print(f"Watertight: {diag.watertight}, volume: {diag.volume}, source: {diag.volume_source}")

    # Voxelization only
    vox = mesh.voxelized(voxel_pitch)
    vox_filled = vox.fill()
    fluid_mask = vox_filled.matrix.astype(bool)
    num_fluid = int(fluid_mask.sum())
    print(f"Num fluid voxels at pitch={voxel_pitch}: {num_fluid}")

    if num_fluid == 0:
        print("[INFO] No fluid voxels – skeletonization not attempted.")
        return

    try:
        skeleton = skeletonize(fluid_mask.astype(bool))
    except Exception as e:
        print("[ERROR] Skeletonization raised an exception:", e)
        return

    num_skel_voxels = int(skeleton.sum())
    print(f"Num skeleton voxels: {num_skel_voxels}")
    if num_skel_voxels == 0:
        print("[WARN] Skeletonization produced an empty skeleton.")
    else:
        print("[OK] Skeletonization produced a non-empty skeleton.")


demo_skeleton_failure("C:/Users/Erick/Downloads/generated_stls/flat_sheet.stl", 1.0)


# In[ ]:





# In[ ]:




