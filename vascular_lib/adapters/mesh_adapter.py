"""
Adapter for converting VascularNetwork to trimesh.Trimesh.

Provides STL mesh export with fast and robust modes.
"""

import numpy as np
import trimesh
from typing import Optional, Literal
from ..core.network import VascularNetwork
from ..core.result import OperationResult, OperationStatus, ErrorCode


def to_trimesh(
    network: VascularNetwork,
    mode: Literal["fast", "robust"] = "fast",
    radial_resolution: int = 8,
    include_caps: bool = True,
    min_segment_length: float = 1e-6,
) -> OperationResult:
    """
    Convert VascularNetwork to trimesh.Trimesh for STL export.
    
    Two modes:
    - "fast": Concatenate all segment meshes, then repair with voxel remesh
    - "robust": Attempt boolean union (requires backend), fallback to fast
    
    Parameters
    ----------
    network : VascularNetwork
        The vascular network to convert
    mode : {"fast", "robust"}
        Export mode
    radial_resolution : int
        Number of vertices around each cylinder
    include_caps : bool
        Whether to include hemispherical caps at segment ends
    min_segment_length : float
        Skip segments shorter than this
        
    Returns
    -------
    OperationResult
        Result with mesh in metadata['mesh']
    """
    try:
        meshes = []
        
        for seg_id, segment in network.segments.items():
            start = np.array([
                segment.geometry.start.x,
                segment.geometry.start.y,
                segment.geometry.start.z,
            ])
            end = np.array([
                segment.geometry.end.x,
                segment.geometry.end.y,
                segment.geometry.end.z,
            ])
            
            length = np.linalg.norm(end - start)
            if length < min_segment_length:
                continue
            
            r_start = segment.geometry.radius_start
            r_end = segment.geometry.radius_end
            
            mesh = _create_capsule_mesh(
                start, end, r_start, r_end,
                radial_resolution=radial_resolution,
                include_caps=include_caps,
            )
            meshes.append(mesh)
        
        for node_id, node in network.nodes.items():
            # Start from 0 so we only grow based on attached segments
            max_radius = 0.0
            for seg in network.segments.values():
                if seg.start_node_id == node_id:
                    max_radius = max(max_radius, seg.geometry.radius_start)
                elif seg.end_node_id == node_id:
                    max_radius = max(max_radius, seg.geometry.radius_end)
        
            # If no segments touch this node, skip making a sphere
            if max_radius <= 0.00002:
                continue
        
            # Make node spheres much smaller than vessel radii
            sphere_radius = max_radius * 0.2  # try 0.1–0.3 depending how subtle you want it

            
            center = np.array([node.position.x, node.position.y, node.position.z])
            sphere = trimesh.creation.icosphere(subdivisions=2, radius=sphere_radius)
            sphere.apply_translation(center)
            meshes.append(sphere)
        
        if not meshes:
            return OperationResult.failure(
                "No meshes created",
                error_codes=[ErrorCode.MESH_EXPORT_FAILED.value],
            )
        
        if mode == "robust":
            try:
                combined = meshes[0]
                for mesh in meshes[1:]:
                    combined = trimesh.boolean.union([combined, mesh], engine='blender')
                
                result = OperationResult.success(
                    f"Robust mesh export: {len(meshes)} components unioned",
                    metadata={
                        'mesh': combined,
                        'mode': 'robust',
                        'num_components': len(meshes),
                        'is_watertight': combined.is_watertight,
                    },
                )
                return result
                
            except Exception as e:
                result = OperationResult.partial_success(
                    f"Boolean union failed, falling back to fast mode: {e}",
                )
                result.add_warning(f"Robust mode failed: {e}")
                mode = "fast"
        
        combined = trimesh.util.concatenate(meshes)
        
        result = OperationResult.success(
            f"Fast mesh export: {len(meshes)} components concatenated",
            metadata={
                'mesh': combined,
                'mode': mode,
                'num_components': len(meshes),
                'is_watertight': combined.is_watertight,
                'needs_repair': not combined.is_watertight,
            },
        )
        
        return result
        
    except Exception as e:
        return OperationResult.failure(
            f"Mesh export failed: {e}",
            error_codes=[ErrorCode.MESH_EXPORT_FAILED.value],
        )


def _create_capsule_mesh(
    start: np.ndarray,
    end: np.ndarray,
    r_start: float,
    r_end: float,
    radial_resolution: int = 8,
    include_caps: bool = True,
) -> trimesh.Trimesh:
    """
    Create a capsule mesh (cylinder with optional hemispherical caps).
    
    Parameters
    ----------
    start : np.ndarray
        Start point [x, y, z]
    end : np.ndarray
        End point [x, y, z]
    r_start : float
        Radius at start
    r_end : float
        Radius at end
    radial_resolution : int
        Number of vertices around cylinder
    include_caps : bool
        Whether to include hemispherical caps
        
    Returns
    -------
    trimesh.Trimesh
        Capsule mesh
    """
    direction = end - start
    length = np.linalg.norm(direction)
    direction = direction / length
    
    cylinder = trimesh.creation.cylinder(
        radius=r_start,  # Use start radius (could interpolate)
        height=length,
        sections=radial_resolution,
    )
    
    z_axis = np.array([0, 0, 1])
    if not np.allclose(direction, z_axis):
        rotation_axis = np.cross(z_axis, direction)
        if np.linalg.norm(rotation_axis) > 1e-6:
            rotation_axis = rotation_axis / np.linalg.norm(rotation_axis)
            angle = np.arccos(np.clip(np.dot(z_axis, direction), -1, 1))
            rotation_matrix = trimesh.transformations.rotation_matrix(
                angle, rotation_axis
            )
            cylinder.apply_transform(rotation_matrix)
    
    center = (start + end) / 2
    cylinder.apply_translation(center)
    
    if include_caps:
        cap_start = trimesh.creation.icosphere(subdivisions=1, radius=r_start)
        cap_start.apply_translation(start)
        
        cap_end = trimesh.creation.icosphere(subdivisions=1, radius=r_end)
        cap_end.apply_translation(end)
        
        return trimesh.util.concatenate([cylinder, cap_start, cap_end])
    
    return cylinder


def export_stl(
    network: VascularNetwork,
    output_path: str,
    mode: Literal["fast", "robust"] = "fast",
    repair: bool = True,
    **kwargs,
) -> OperationResult:
    """
    Export VascularNetwork to STL file.
    
    Parameters
    ----------
    network : VascularNetwork
        The vascular network to export
    output_path : str
        Path to output STL file
    mode : {"fast", "robust"}
        Export mode
    repair : bool
        Whether to repair mesh before export
    **kwargs
        Additional arguments passed to to_trimesh
        
    Returns
    -------
    OperationResult
        Result of export operation
    """
    result = to_trimesh(network, mode=mode, **kwargs)
    
    if not result.is_success():
        return result
    
    mesh = result.metadata['mesh']
    
    if repair and not mesh.is_watertight:
        try:
            from vascular_network.mesh.repair import meshfix_repair
            mesh = meshfix_repair(mesh, keep_largest_component=True)
            result.add_warning("Mesh repaired with meshfix")
            result.metadata['was_repaired'] = True
            result.metadata['is_watertight'] = mesh.is_watertight
        except Exception as e:
            result.add_warning(f"Mesh repair failed: {e}")
            result.metadata['repair_failed'] = True
    
    try:
        mesh.export(output_path)
        result.message = f"Exported to {output_path}"
        result.metadata['output_path'] = output_path
        return result
    except Exception as e:
        return OperationResult.failure(
            f"Export failed: {e}",
            error_codes=[ErrorCode.MESH_EXPORT_FAILED.value],
        )
