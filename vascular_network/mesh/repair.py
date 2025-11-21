import numpy as np
import trimesh
from pymeshfix import MeshFix
from scipy import ndimage
from skimage.measure import marching_cubes
import trimesh.smoothing as tmsmooth


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

    pitch_eff = auto_adjust_voxel_pitch(
        mesh, requested_pitch=pitch, max_voxels=max_voxels
    )

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

    structure = ndimage.generate_binary_structure(3, 1)
    labels, num_labels = ndimage.label(fluid_mask, structure=structure)
    if num_labels > 1:
        sizes = ndimage.sum(fluid_mask, labels, index=range(1, num_labels + 1))
        largest_label = 1 + int(np.argmax(sizes))
        fluid_mask = labels == largest_label

    if closing_iters > 0:
        fluid_mask = ndimage.binary_closing(fluid_mask, iterations=closing_iters)

    if opening_iters > 0:
        fluid_mask = ndimage.binary_opening(fluid_mask, iterations=opening_iters)

    if dilation_iters > 0:
        fluid_mask = ndimage.binary_dilation(fluid_mask, iterations=dilation_iters)

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
        return mesh

    raw_scale = (target_volume / vol_now) ** (1.0 / 3.0)

    min_s = 1.0 / max_scale_factor
    max_s = max_scale_factor
    scale = float(np.clip(raw_scale, min_s, max_s))

    center = mesh.center_mass
    mesh.vertices -= center
    mesh.vertices *= scale
    mesh.vertices += center

    mesh.remove_unreferenced_vertices()
    return mesh


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
        joincomp=True,
        remove_smallest_components=False,
    )

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

    if keep_largest_component:
        comps = repaired.split(only_watertight=False)
        if len(comps) > 1:
            largest = max(comps, key=lambda c: c.faces.shape[0])
            repaired = largest.copy()
            repaired.remove_unreferenced_vertices()

    try:
        trimesh.repair.fix_normals(repaired, multibody=True)
        trimesh.repair.fix_winding(repaired)
    except Exception:
        pass

    repaired.remove_unreferenced_vertices()
    return repaired
