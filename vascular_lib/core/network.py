"""
Core network data structures.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any
from .types import Point3D, TubeGeometry
from .ids import IDGenerator
from .domain import DomainSpec


@dataclass
class Node:
    """
    Node in a vascular network.
    
    Represents junctions, inlets, outlets, or terminal points.
    """
    
    id: int
    position: Point3D
    node_type: str  # "junction", "inlet", "outlet", "terminal"
    vessel_type: str  # "arterial", "venous"
    attributes: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        return {
            "id": self.id,
            "position": self.position.to_dict(),
            "node_type": self.node_type,
            "vessel_type": self.vessel_type,
            "attributes": self.attributes,
        }
    
    @classmethod
    def from_dict(cls, d: dict) -> "Node":
        """Create from dictionary."""
        return cls(
            id=d["id"],
            position=Point3D.from_dict(d["position"]),
            node_type=d["node_type"],
            vessel_type=d["vessel_type"],
            attributes=d.get("attributes", {}),
        )


@dataclass
class VesselSegment:
    """
    Vessel segment connecting two nodes.
    
    Represents a tubular blood vessel with geometry and properties.
    """
    
    id: int
    start_node_id: int
    end_node_id: int
    geometry: TubeGeometry
    vessel_type: str  # "arterial", "venous"
    attributes: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        return {
            "id": self.id,
            "start_node_id": self.start_node_id,
            "end_node_id": self.end_node_id,
            "geometry": self.geometry.to_dict(),
            "vessel_type": self.vessel_type,
            "attributes": self.attributes,
        }
    
    @classmethod
    def from_dict(cls, d: dict) -> "VesselSegment":
        """Create from dictionary."""
        return cls(
            id=d["id"],
            start_node_id=d["start_node_id"],
            end_node_id=d["end_node_id"],
            geometry=TubeGeometry.from_dict(d["geometry"]),
            vessel_type=d["vessel_type"],
            attributes=d.get("attributes", {}),
        )


class VascularNetwork:
    """
    Complete vascular network with nodes, segments, and domain.
    
    This is the main data structure for LLM-driven vascular design.
    """
    
    def __init__(
        self,
        domain: DomainSpec,
        metadata: Optional[Dict[str, Any]] = None,
        seed: Optional[int] = None,
    ):
        """
        Initialize vascular network.
        
        Parameters
        ----------
        domain : DomainSpec
            Geometric domain for the network
        metadata : dict, optional
            Network metadata (name, organ type, units, etc.)
        seed : int, optional
            Random seed for deterministic operations
        """
        self.nodes: Dict[int, Node] = {}
        self.segments: Dict[int, VesselSegment] = {}
        self.domain = domain
        self.metadata = metadata or {}
        self.id_gen = IDGenerator(seed=seed)
        
        self._spatial_index = None
        
        self._history: List[Any] = []
        self._history_index = -1
    
    def add_node(self, node: Node) -> None:
        """Add a node to the network."""
        self.nodes[node.id] = node
        self._invalidate_spatial_index()
    
    def add_segment(self, segment: VesselSegment) -> None:
        """Add a segment to the network."""
        if segment.start_node_id not in self.nodes:
            raise ValueError(f"Start node {segment.start_node_id} not in network")
        if segment.end_node_id not in self.nodes:
            raise ValueError(f"End node {segment.end_node_id} not in network")
        self.segments[segment.id] = segment
        self._invalidate_spatial_index()
    
    def remove_node(self, node_id: int) -> None:
        """Remove a node from the network."""
        if node_id in self.nodes:
            del self.nodes[node_id]
            self._invalidate_spatial_index()
    
    def remove_segment(self, segment_id: int) -> None:
        """Remove a segment from the network."""
        if segment_id in self.segments:
            del self.segments[segment_id]
            self._invalidate_spatial_index()
    
    def get_node(self, node_id: int) -> Optional[Node]:
        """Get node by ID."""
        return self.nodes.get(node_id)
    
    def get_segment(self, segment_id: int) -> Optional[VesselSegment]:
        """Get segment by ID."""
        return self.segments.get(segment_id)
    
    def get_spatial_index(self):
        """Get or create spatial index."""
        if self._spatial_index is None:
            from ..spatial.grid_index import SpatialIndex
            self._spatial_index = SpatialIndex(self)
        return self._spatial_index
    
    def _invalidate_spatial_index(self) -> None:
        """Invalidate spatial index after modifications."""
        self._spatial_index = None
    
    def snapshot(self) -> dict:
        """Create a snapshot of current network state."""
        return {
            "nodes": {nid: node.to_dict() for nid, node in self.nodes.items()},
            "segments": {sid: seg.to_dict() for sid, seg in self.segments.items()},
            "domain": self.domain.to_dict(),
            "metadata": self.metadata.copy(),
            "id_gen_state": self.id_gen.get_state(),
        }
    
    def restore(self, snapshot: dict) -> None:
        """Restore network from a snapshot."""
        self.nodes = {int(nid): Node.from_dict(n) for nid, n in snapshot["nodes"].items()}
        self.segments = {int(sid): VesselSegment.from_dict(s) for sid, s in snapshot["segments"].items()}
        self.metadata = snapshot["metadata"].copy()
        self.id_gen.set_state(snapshot["id_gen_state"])
        self._invalidate_spatial_index()
    
    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        return {
            "schema_version": "1.0",
            "nodes": {nid: node.to_dict() for nid, node in self.nodes.items()},
            "segments": {sid: seg.to_dict() for sid, seg in self.segments.items()},
            "domain": self.domain.to_dict(),
            "metadata": self.metadata,
            "id_gen_state": self.id_gen.get_state(),
        }
    
    @classmethod
    def from_dict(cls, d: dict) -> "VascularNetwork":
        """Create from dictionary."""
        from .domain import domain_from_dict
        
        domain = domain_from_dict(d["domain"])
        network = cls(domain=domain, metadata=d.get("metadata", {}))
        
        for nid, node_dict in d["nodes"].items():
            network.nodes[int(nid)] = Node.from_dict(node_dict)
        
        for sid, seg_dict in d["segments"].items():
            network.segments[int(sid)] = VesselSegment.from_dict(seg_dict)
        
        if "id_gen_state" in d:
            network.id_gen.set_state(d["id_gen_state"])
        
        return network
