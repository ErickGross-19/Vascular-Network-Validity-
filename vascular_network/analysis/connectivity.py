import numpy as np
import trimesh
from scipy import ndimage
from typing import Tuple, Dict, Any


def mesh_to_fluid_mask(mesh, pitch):
    """
    Convert a mesh to a voxelized fluid mask.
    
    Returns
    -------
    fluid_mask : np.ndarray
        Boolean array of voxel occupancy
    origin : np.ndarray
        Origin of the voxel grid
    pitch : float
        Voxel pitch (spacing)
    """
    vox = mesh.voxelized(pitch)
    vox_filled = vox.fill()

    fluid_mask = vox_filled.matrix.astype(bool)

    if hasattr(vox_filled, "origin"):
        origin = np.array(vox_filled.origin, dtype=float)
    else:
        origin = np.array(vox_filled.transform[:3, 3], dtype=float)

    pitch_attr = getattr(vox_filled, "pitch", pitch)
    pitch_arr = np.asarray(pitch_attr, dtype=float)

    if pitch_arr.size == 1:
        pitch_val = float(pitch_arr)
    else:
        pitch_val = float(pitch_arr[0])

    return fluid_mask, origin, pitch_val


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

    structure = ndimage.generate_binary_structure(rank=3, connectivity=1)
    labels, num_labels = ndimage.label(fluid_mask, structure=structure)

    component_sizes = ndimage.sum(fluid_mask, labels, index=range(1, num_labels + 1))
    component_sizes = [int(s) for s in component_sizes]

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
