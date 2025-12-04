"""
Negative space embedding operations for vascular networks.

This module provides functionality to embed vascular tree STL meshes into
domain volumes (box or ellipsoid) as negative space (voids).
"""

import numpy as np
import trimesh
from pathlib import Path
from typing import Optional, Dict, Tuple, Union
from scipy import ndimage
from skimage.measure import marching_cubes

from ..core.domain import DomainSpec, BoxDomain, EllipsoidDomain


def embed_tree_as_negative_space(
    tree_stl_path: Union[str, Path],
    domain: DomainSpec,
    voxel_pitch: float = 0.001,
    margin: float = 0.0,
    dilation_voxels: int = 0,
    smoothing_iters: int = 5,
    output_void: bool = True,
    output_shell: bool = False,
    shell_thickness: float = 0.002,
) -> Dict[str, Optional[trimesh.Trimesh]]:
    """
    Embed a vascular tree STL mesh into a domain as negative space (void).
    
    This creates a solid domain mesh with the tree carved out as a void.
    Useful for creating molds, scaffolds, or perfusion chambers.
    
    Parameters
    ----------
    tree_stl_path : str or Path
        Path to the vascular tree STL file
    domain : DomainSpec
        Domain specification (BoxDomain or EllipsoidDomain)
    voxel_pitch : float
        Voxel size in world units (default: 0.001 = 1mm)
        Smaller values give higher resolution but slower computation
    margin : float
        Additional margin around tree bounds when auto-sizing domain (default: 0)
    dilation_voxels : int
        Number of voxels to dilate the tree by (useful if tree is thin centerlines)
        Default: 0 (no dilation)
    smoothing_iters : int
        Number of smoothing iterations to reduce voxel artifacts (default: 5)
    output_void : bool
        Whether to output the void mesh (tree volume) (default: True)
    output_shell : bool
        Whether to output a shell mesh around the void (default: False)
    shell_thickness : float
        Thickness of shell around void in world units (default: 0.002 = 2mm)
    
    Returns
    -------
    dict
        Dictionary with keys:
        - 'domain_with_void': trimesh.Trimesh of solid domain with void carved out
        - 'void': trimesh.Trimesh of the void volume (if output_void=True)
        - 'shell': trimesh.Trimesh of shell around void (if output_shell=True)
        - 'metadata': dict with voxel grid info and statistics
    
    Examples
    --------
    >>> from vascular_lib.core.domain import BoxDomain
    >>> from vascular_lib.core.types import Point3D
    >>> from vascular_lib.ops.embedding import embed_tree_as_negative_space
    >>> 
    >>> # Create a box domain
    >>> domain = BoxDomain.from_center_and_size(
    ...     center=Point3D(0, 0, 0),
    ...     width=0.1, height=0.1, depth=0.1  # 100mm cube
    ... )
    >>> 
    >>> # Embed tree as negative space
    >>> result = embed_tree_as_negative_space(
    ...     tree_stl_path='tree.stl',
    ...     domain=domain,
    ...     voxel_pitch=0.0005,  # 0.5mm voxels
    ...     output_void=True,
    ...     output_shell=True
    ... )
    >>> 
    >>> # Export results
    >>> result['domain_with_void'].export('domain_with_void.stl')
    >>> result['void'].export('void.stl')
    >>> result['shell'].export('shell.stl')
    """
    tree_stl_path = Path(tree_stl_path)
    
    tree_mesh = trimesh.load(tree_stl_path)
    if not isinstance(tree_mesh, trimesh.Trimesh):
        raise ValueError(f"Expected Trimesh object, got {type(tree_mesh)}")
    
    tree_bounds = tree_mesh.bounds  # [[xmin, ymin, zmin], [xmax, ymax, zmax]]
    tree_min = tree_bounds[0]
    tree_max = tree_bounds[1]
    tree_center = (tree_min + tree_max) / 2
    tree_size = tree_max - tree_min
    
    if isinstance(domain, BoxDomain):
        domain_min = np.array([domain.x_min, domain.y_min, domain.z_min])
        domain_max = np.array([domain.x_max, domain.y_max, domain.z_max])
    elif isinstance(domain, EllipsoidDomain):
        center = np.array([domain.center_x, domain.center_y, domain.center_z])
        radii = np.array([domain.radius_x, domain.radius_y, domain.radius_z])
        domain_min = center - radii
        domain_max = center + radii
    else:
        raise ValueError(f"Unsupported domain type: {type(domain)}")
    
    if margin > 0:
        domain_min -= margin
        domain_max += margin
    
    grid_shape = np.ceil((domain_max - domain_min) / voxel_pitch).astype(int)
    
    grid_shape = np.maximum(grid_shape, 10)
    
    padding = 2
    grid_shape_padded = grid_shape + 2 * padding
    domain_min_padded = domain_min - padding * voxel_pitch
    
    print(f"Creating voxel grid: {grid_shape_padded} voxels (with {padding}-voxel padding)")
    print(f"Domain bounds: {domain_min} to {domain_max}")
    print(f"Padded bounds: {domain_min_padded} to {domain_min_padded + grid_shape_padded * voxel_pitch}")
    print(f"Voxel pitch: {voxel_pitch}")
    
    domain_mask = np.zeros(grid_shape_padded, dtype=bool)
    
    if isinstance(domain, BoxDomain):
        domain_mask[padding:-padding, padding:-padding, padding:-padding] = True
    elif isinstance(domain, EllipsoidDomain):
        center = np.array([domain.center.x, domain.center.y, domain.center.z])
        radii = np.array([domain.semi_axis_a, domain.semi_axis_b, domain.semi_axis_c])
        
        x = np.linspace(domain_min_padded[0], domain_min_padded[0] + grid_shape_padded[0] * voxel_pitch, grid_shape_padded[0])
        y = np.linspace(domain_min_padded[1], domain_min_padded[1] + grid_shape_padded[1] * voxel_pitch, grid_shape_padded[1])
        z = np.linspace(domain_min_padded[2], domain_min_padded[2] + grid_shape_padded[2] * voxel_pitch, grid_shape_padded[2])
        
        xx, yy, zz = np.meshgrid(x, y, z, indexing='ij')
        
        ellipsoid_eq = (
            ((xx - center[0]) / radii[0]) ** 2 +
            ((yy - center[1]) / radii[1]) ** 2 +
            ((zz - center[2]) / radii[2]) ** 2
        )
        domain_mask = ellipsoid_eq <= 1.0
    
    print(f"Domain mask: {domain_mask.sum()} voxels ({100 * domain_mask.sum() / domain_mask.size:.1f}%)")
    
    tree_voxels = tree_mesh.voxelized(pitch=voxel_pitch)
    tree_mask = tree_voxels.matrix
    
    tree_origin = tree_voxels.transform[:3, 3]
    
    offset_voxels = np.round((tree_origin - domain_min_padded) / voxel_pitch).astype(int)
    
    aligned_tree_mask = np.zeros(grid_shape_padded, dtype=bool)
    
    tree_shape = np.array(tree_mask.shape)
    
    src_start = np.maximum(-offset_voxels, 0)
    dst_start = np.maximum(offset_voxels, 0)
    
    copy_size = np.minimum(
        tree_shape - src_start,  # Remaining tree voxels from src_start
        grid_shape_padded - dst_start   # Remaining domain voxels from dst_start
    )
    
    copy_size = np.maximum(copy_size, 0)
    
    if np.all(copy_size > 0):
        aligned_tree_mask[
            dst_start[0]:dst_start[0] + copy_size[0],
            dst_start[1]:dst_start[1] + copy_size[1],
            dst_start[2]:dst_start[2] + copy_size[2]
        ] = tree_mask[
            src_start[0]:src_start[0] + copy_size[0],
            src_start[1]:src_start[1] + copy_size[1],
            src_start[2]:src_start[2] + copy_size[2]
        ]
    
    print(f"Tree mask: {aligned_tree_mask.sum()} voxels ({100 * aligned_tree_mask.sum() / aligned_tree_mask.size:.1f}%)")
    
    if dilation_voxels > 0:
        print(f"Dilating tree by {dilation_voxels} voxels...")
        aligned_tree_mask = ndimage.binary_dilation(
            aligned_tree_mask,
            iterations=dilation_voxels
        )
        print(f"After dilation: {aligned_tree_mask.sum()} voxels")
    
    void_mask = aligned_tree_mask & domain_mask
    
    solid_mask = domain_mask & (~void_mask)
    
    print(f"Void mask: {void_mask.sum()} voxels")
    print(f"Solid mask: {solid_mask.sum()} voxels")
    
    if smoothing_iters > 0:
        print(f"Smoothing with {smoothing_iters} iterations...")
        solid_mask = ndimage.binary_closing(solid_mask, iterations=smoothing_iters // 2)
        solid_mask = ndimage.binary_opening(solid_mask, iterations=smoothing_iters // 2)
    
    result = {}
    
    if solid_mask.any():
        print("Generating domain mesh with marching cubes...")
        verts, faces, _, _ = marching_cubes(
            volume=solid_mask.astype(np.uint8),
            level=0.5,
            spacing=(voxel_pitch, voxel_pitch, voxel_pitch),
            allow_degenerate=False,
        )
        
        verts = verts[:, [2, 1, 0]]
        
        verts += domain_min_padded
        
        domain_mesh = trimesh.Trimesh(
            vertices=verts,
            faces=faces.astype(np.int64),
            process=False,
        )
        
        domain_mesh.remove_unreferenced_vertices()
        
        if domain_mesh.volume < 0:
            domain_mesh.invert()
        
        trimesh.repair.fix_normals(domain_mesh)
        
        if not domain_mesh.is_watertight:
            trimesh.repair.fill_holes(domain_mesh)
        
        result['domain_with_void'] = domain_mesh
        print(f"Domain mesh: {len(domain_mesh.vertices)} vertices, {len(domain_mesh.faces)} faces")
        print(f"  Watertight: {domain_mesh.is_watertight}, Volume: {domain_mesh.volume:.9f}")
    else:
        result['domain_with_void'] = None
        print("Warning: Solid mask is empty, no domain mesh generated")
    
    if output_void and void_mask.any():
        print("Generating void mesh with marching cubes...")
        verts, faces, _, _ = marching_cubes(
            volume=void_mask.astype(np.uint8),
            level=0.5,
            spacing=(voxel_pitch, voxel_pitch, voxel_pitch),
            allow_degenerate=False,
        )
        
        verts = verts[:, [2, 1, 0]]
        
        verts += domain_min_padded
        
        void_mesh = trimesh.Trimesh(
            vertices=verts,
            faces=faces.astype(np.int64),
            process=False,
        )
        
        void_mesh.remove_unreferenced_vertices()
        
        if void_mesh.volume < 0:
            void_mesh.invert()
        
        trimesh.repair.fix_normals(void_mesh)
        
        if not void_mesh.is_watertight:
            trimesh.repair.fill_holes(void_mesh)
        
        result['void'] = void_mesh
        print(f"Void mesh: {len(void_mesh.vertices)} vertices, {len(void_mesh.faces)} faces")
        print(f"  Watertight: {void_mesh.is_watertight}, Volume: {void_mesh.volume:.9f}")
    else:
        result['void'] = None
    
    if output_shell and void_mask.any():
        print(f"Generating shell mesh (thickness={shell_thickness})...")
        dist_from_void = ndimage.distance_transform_edt(~void_mask) * voxel_pitch
        
        shell_mask = (dist_from_void <= shell_thickness) & domain_mask & (~void_mask)
        
        if shell_mask.any():
            verts, faces, _, _ = marching_cubes(
                volume=shell_mask.astype(np.uint8),
                level=0.5,
                spacing=(voxel_pitch, voxel_pitch, voxel_pitch),
                allow_degenerate=False,
            )
            
            verts = verts[:, [2, 1, 0]]
            
            verts += domain_min_padded
            
            shell_mesh = trimesh.Trimesh(
                vertices=verts,
                faces=faces.astype(np.int64),
                process=False,
            )
            
            shell_mesh.remove_unreferenced_vertices()
            
            if shell_mesh.volume < 0:
                shell_mesh.invert()
            
            trimesh.repair.fix_normals(shell_mesh)
            
            if not shell_mesh.is_watertight:
                trimesh.repair.fill_holes(shell_mesh)
            
            result['shell'] = shell_mesh
            print(f"Shell mesh: {len(shell_mesh.vertices)} vertices, {len(shell_mesh.faces)} faces")
            print(f"  Watertight: {shell_mesh.is_watertight}, Volume: {shell_mesh.volume:.9f}")
        else:
            result['shell'] = None
            print("Warning: Shell mask is empty, no shell mesh generated")
    else:
        result['shell'] = None
    
    result['metadata'] = {
        'voxel_pitch': voxel_pitch,
        'grid_shape': grid_shape_padded.tolist(),
        'domain_bounds': {
            'min': domain_min.tolist(),
            'max': domain_max.tolist(),
        },
        'padded_bounds': {
            'min': domain_min_padded.tolist(),
            'max': (domain_min_padded + grid_shape_padded * voxel_pitch).tolist(),
        },
        'tree_bounds': {
            'min': tree_min.tolist(),
            'max': tree_max.tolist(),
        },
        'voxel_counts': {
            'domain': int(domain_mask.sum()),
            'tree': int(aligned_tree_mask.sum()),
            'void': int(void_mask.sum()),
            'solid': int(solid_mask.sum()),
        },
        'dilation_voxels': dilation_voxels,
        'smoothing_iters': smoothing_iters,
    }
    
    return result
