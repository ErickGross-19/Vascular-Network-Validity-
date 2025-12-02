"""
Geometric domain specifications for vascular networks.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional
import numpy as np
from .types import Point3D


class DomainSpec(ABC):
    """Abstract base class for geometric domains."""
    
    @abstractmethod
    def contains(self, point: Point3D) -> bool:
        """Check if a point is inside the domain."""
        pass
    
    @abstractmethod
    def project_inside(self, point: Point3D) -> Point3D:
        """Project a point to the nearest point inside the domain."""
        pass
    
    @abstractmethod
    def distance_to_boundary(self, point: Point3D) -> float:
        """Compute distance from point to domain boundary."""
        pass
    
    @abstractmethod
    def sample_points(self, n_points: int, seed: Optional[int] = None) -> np.ndarray:
        """Sample random points inside the domain."""
        pass
    
    @abstractmethod
    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        pass
    
    @abstractmethod
    def get_bounds(self) -> tuple:
        """Get bounding box (min_x, max_x, min_y, max_y, min_z, max_z)."""
        pass


@dataclass
class EllipsoidDomain(DomainSpec):
    """Ellipsoidal domain (e.g., for liver)."""
    
    semi_axis_a: float  # x-axis
    semi_axis_b: float  # y-axis
    semi_axis_c: float  # z-axis
    center: Point3D = None
    
    def __post_init__(self):
        if self.center is None:
            self.center = Point3D(0.0, 0.0, 0.0)
    
    def contains(self, point: Point3D) -> bool:
        """Check if point is inside ellipsoid."""
        dx = point.x - self.center.x
        dy = point.y - self.center.y
        dz = point.z - self.center.z
        
        normalized = (
            (dx / self.semi_axis_a) ** 2 +
            (dy / self.semi_axis_b) ** 2 +
            (dz / self.semi_axis_c) ** 2
        )
        return normalized <= 1.0
    
    def project_inside(self, point: Point3D) -> Point3D:
        """Project point to nearest point inside ellipsoid."""
        if self.contains(point):
            return point
        
        dx = point.x - self.center.x
        dy = point.y - self.center.y
        dz = point.z - self.center.z
        
        r = np.sqrt(dx**2 + dy**2 + dz**2)
        if r < 1e-10:
            return self.center
        
        direction = np.array([dx / r, dy / r, dz / r])
        
        t = 1.0 / np.sqrt(
            (direction[0] / self.semi_axis_a) ** 2 +
            (direction[1] / self.semi_axis_b) ** 2 +
            (direction[2] / self.semi_axis_c) ** 2
        )
        
        t *= 0.99
        
        return Point3D(
            self.center.x + t * direction[0],
            self.center.y + t * direction[1],
            self.center.z + t * direction[2],
        )
    
    def distance_to_boundary(self, point: Point3D) -> float:
        """Approximate distance to boundary."""
        dx = point.x - self.center.x
        dy = point.y - self.center.y
        dz = point.z - self.center.z
        
        r_local = np.sqrt(dx**2 + dy**2 + dz**2)
        if r_local < 1e-10:
            return min(self.semi_axis_a, self.semi_axis_b, self.semi_axis_c)
        
        direction = np.array([dx / r_local, dy / r_local, dz / r_local])
        
        t_surface = 1.0 / np.sqrt(
            (direction[0] / self.semi_axis_a) ** 2 +
            (direction[1] / self.semi_axis_b) ** 2 +
            (direction[2] / self.semi_axis_c) ** 2
        )
        
        surface_point = np.array([
            self.center.x + t_surface * direction[0],
            self.center.y + t_surface * direction[1],
            self.center.z + t_surface * direction[2],
        ])
        
        point_arr = point.to_array()
        return float(np.linalg.norm(surface_point - point_arr))
    
    def sample_points(self, n_points: int, seed: Optional[int] = None) -> np.ndarray:
        """Sample random points uniformly inside ellipsoid."""
        rng = np.random.default_rng(seed)
        
        points = []
        while len(points) < n_points:
            x = rng.uniform(-self.semi_axis_a, self.semi_axis_a)
            y = rng.uniform(-self.semi_axis_b, self.semi_axis_b)
            z = rng.uniform(-self.semi_axis_c, self.semi_axis_c)
            
            point = Point3D(
                self.center.x + x,
                self.center.y + y,
                self.center.z + z,
            )
            
            if self.contains(point):
                points.append([point.x, point.y, point.z])
        
        return np.array(points)
    
    def get_bounds(self) -> tuple:
        """Get bounding box."""
        return (
            self.center.x - self.semi_axis_a,
            self.center.x + self.semi_axis_a,
            self.center.y - self.semi_axis_b,
            self.center.y + self.semi_axis_b,
            self.center.z - self.semi_axis_c,
            self.center.z + self.semi_axis_c,
        )
    
    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "type": "ellipsoid",
            "semi_axis_a": self.semi_axis_a,
            "semi_axis_b": self.semi_axis_b,
            "semi_axis_c": self.semi_axis_c,
            "center": self.center.to_dict(),
        }
    
    @classmethod
    def from_dict(cls, d: dict) -> "EllipsoidDomain":
        """Create from dictionary."""
        return cls(
            semi_axis_a=d["semi_axis_a"],
            semi_axis_b=d["semi_axis_b"],
            semi_axis_c=d["semi_axis_c"],
            center=Point3D.from_dict(d["center"]),
        )


@dataclass
class MeshDomain(DomainSpec):
    """Mesh-based domain from STL file."""
    
    mesh_path: str
    _mesh: Optional[object] = None
    
    def __post_init__(self):
        """Load mesh on initialization."""
        try:
            import trimesh
            self._mesh = trimesh.load(self.mesh_path)
        except ImportError:
            raise ImportError("trimesh is required for MeshDomain. Install with: pip install trimesh")
        except Exception as e:
            raise ValueError(f"Failed to load mesh from {self.mesh_path}: {e}")
    
    def contains(self, point: Point3D) -> bool:
        """Check if point is inside mesh."""
        point_arr = point.to_array().reshape(1, 3)
        return bool(self._mesh.contains(point_arr)[0])
    
    def project_inside(self, point: Point3D) -> Point3D:
        """Project point to nearest point inside mesh."""
        if self.contains(point):
            return point
        
        point_arr = point.to_array().reshape(1, 3)
        closest, distance, triangle_id = self._mesh.nearest.on_surface(point_arr)
        
        normal = self._mesh.face_normals[triangle_id[0]]
        
        offset = -0.001 * normal
        inside_point = closest[0] + offset
        
        return Point3D.from_array(inside_point)
    
    def distance_to_boundary(self, point: Point3D) -> float:
        """Compute distance to mesh boundary."""
        point_arr = point.to_array().reshape(1, 3)
        closest, distance, triangle_id = self._mesh.nearest.on_surface(point_arr)
        return float(distance[0])
    
    def sample_points(self, n_points: int, seed: Optional[int] = None) -> np.ndarray:
        """Sample random points inside mesh."""
        rng = np.random.default_rng(seed)
        
        bounds = self._mesh.bounds
        min_bound = bounds[0]
        max_bound = bounds[1]
        
        points = []
        max_attempts = n_points * 100
        attempts = 0
        
        while len(points) < n_points and attempts < max_attempts:
            x = rng.uniform(min_bound[0], max_bound[0])
            y = rng.uniform(min_bound[1], max_bound[1])
            z = rng.uniform(min_bound[2], max_bound[2])
            
            point = Point3D(x, y, z)
            
            if self.contains(point):
                points.append([x, y, z])
            
            attempts += 1
        
        if len(points) < n_points:
            raise ValueError(f"Could only sample {len(points)} points after {max_attempts} attempts")
        
        return np.array(points)
    
    def get_bounds(self) -> tuple:
        """Get bounding box."""
        bounds = self._mesh.bounds
        return (
            bounds[0][0], bounds[1][0],
            bounds[0][1], bounds[1][1],
            bounds[0][2], bounds[1][2],
        )
    
    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "type": "mesh",
            "mesh_path": self.mesh_path,
        }
    
    @classmethod
    def from_dict(cls, d: dict) -> "MeshDomain":
        """Create from dictionary."""
        return cls(mesh_path=d["mesh_path"])


def domain_from_dict(d: dict) -> DomainSpec:
    """Create domain from dictionary based on type."""
    domain_type = d.get("type")
    
    if domain_type == "ellipsoid":
        return EllipsoidDomain.from_dict(d)
    elif domain_type == "mesh":
        return MeshDomain.from_dict(d)
    else:
        raise ValueError(f"Unknown domain type: {domain_type}")
