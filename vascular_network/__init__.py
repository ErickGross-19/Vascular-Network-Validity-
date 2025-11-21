"""
Vascular Network Validation and Analysis Package

A comprehensive package for validating, repairing, and analyzing vascular network geometries.
Supports STL and Python CAD files as input, with capabilities for:
- Watertight mesh generation
- Connectivity analysis
- Centerline extraction
- CFD (Poiseuille flow) analysis
- Comprehensive visualization and reporting
"""

from .pipeline import validate_and_repair_geometry
from .models import (
    MeshDiagnostics,
    SurfaceQuality,
    ValidationFlags,
    ValidationReport,
)

__version__ = "0.1.0"

__all__ = [
    "validate_and_repair_geometry",
    "MeshDiagnostics",
    "SurfaceQuality",
    "ValidationFlags",
    "ValidationReport",
]
