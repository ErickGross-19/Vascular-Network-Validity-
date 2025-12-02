"""
Operation result types for structured feedback.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional
from enum import Enum


class OperationStatus(Enum):
    """Status of an operation."""
    SUCCESS = "success"
    PARTIAL_SUCCESS = "partial_success"
    FAILURE = "failure"
    WARNING = "warning"


@dataclass
class Delta:
    """
    Record of changes made by an operation.
    
    Used for undo/redo functionality.
    """
    
    created_node_ids: List[int] = field(default_factory=list)
    created_segment_ids: List[int] = field(default_factory=list)
    deleted_node_ids: List[int] = field(default_factory=list)
    deleted_segment_ids: List[int] = field(default_factory=list)
    modified_nodes: Dict[int, dict] = field(default_factory=dict)
    modified_segments: Dict[int, dict] = field(default_factory=dict)
    
    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        return {
            "created_node_ids": self.created_node_ids,
            "created_segment_ids": self.created_segment_ids,
            "deleted_node_ids": self.deleted_node_ids,
            "deleted_segment_ids": self.deleted_segment_ids,
            "modified_nodes": self.modified_nodes,
            "modified_segments": self.modified_segments,
        }
    
    @classmethod
    def from_dict(cls, d: dict) -> "Delta":
        """Create from dictionary."""
        return cls(
            created_node_ids=d.get("created_node_ids", []),
            created_segment_ids=d.get("created_segment_ids", []),
            deleted_node_ids=d.get("deleted_node_ids", []),
            deleted_segment_ids=d.get("deleted_segment_ids", []),
            modified_nodes=d.get("modified_nodes", {}),
            modified_segments=d.get("modified_segments", {}),
        )


@dataclass
class OperationResult:
    """
    Structured result from a vascular network operation.
    
    Provides explicit feedback for LLM consumption.
    """
    
    status: OperationStatus
    message: str = ""
    new_ids: Dict[str, Any] = field(default_factory=dict)
    warnings: List[str] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)
    delta: Optional[Delta] = None
    rng_state: Optional[dict] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def is_success(self) -> bool:
        """Check if operation was successful."""
        return self.status in (OperationStatus.SUCCESS, OperationStatus.PARTIAL_SUCCESS)
    
    def is_failure(self) -> bool:
        """Check if operation failed."""
        return self.status == OperationStatus.FAILURE
    
    def add_warning(self, warning: str) -> None:
        """Add a warning message."""
        self.warnings.append(warning)
    
    def add_error(self, error: str) -> None:
        """Add an error message."""
        self.errors.append(error)
    
    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        return {
            "status": self.status.value,
            "message": self.message,
            "new_ids": self.new_ids,
            "warnings": self.warnings,
            "errors": self.errors,
            "delta": self.delta.to_dict() if self.delta else None,
            "rng_state": self.rng_state,
            "metadata": self.metadata,
        }
    
    @classmethod
    def from_dict(cls, d: dict) -> "OperationResult":
        """Create from dictionary."""
        return cls(
            status=OperationStatus(d["status"]),
            message=d.get("message", ""),
            new_ids=d.get("new_ids", {}),
            warnings=d.get("warnings", []),
            errors=d.get("errors", []),
            delta=Delta.from_dict(d["delta"]) if d.get("delta") else None,
            rng_state=d.get("rng_state"),
            metadata=d.get("metadata", {}),
        )
    
    @classmethod
    def success(cls, message: str = "", **kwargs) -> "OperationResult":
        """Create a success result."""
        return cls(status=OperationStatus.SUCCESS, message=message, **kwargs)
    
    @classmethod
    def failure(cls, message: str = "", **kwargs) -> "OperationResult":
        """Create a failure result."""
        return cls(status=OperationStatus.FAILURE, message=message, **kwargs)
    
    @classmethod
    def partial_success(cls, message: str = "", **kwargs) -> "OperationResult":
        """Create a partial success result."""
        return cls(status=OperationStatus.PARTIAL_SUCCESS, message=message, **kwargs)
