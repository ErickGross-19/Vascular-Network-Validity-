import pytest
import numpy as np
import trimesh
from pathlib import Path
import tempfile


@pytest.fixture
def temp_dir():
    """Create a temporary directory for test outputs."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def straight_cylinder_mesh():
    """Create a straight cylinder mesh for testing."""
    radius = 0.5
    length = 5.0
    
    cylinder = trimesh.creation.cylinder(
        radius=radius,
        height=length,
        sections=32
    )
    
    return cylinder, radius, length


@pytest.fixture
def y_branch_mesh():
    """Create a Y-branch mesh for testing."""
    R_in = 0.5
    L_in = 2.0
    inlet = trimesh.creation.cylinder(radius=R_in, height=L_in, sections=32)
    inlet.apply_translation([0, 0, -L_in/2])
    
    R_out1 = 0.4
    L_out1 = 2.0
    outlet1 = trimesh.creation.cylinder(radius=R_out1, height=L_out1, sections=32)
    outlet1.apply_translation([0, 0, L_out1/2])
    angle1 = np.radians(30)
    rotation1 = trimesh.transformations.rotation_matrix(angle1, [1, 0, 0])
    outlet1.apply_transform(rotation1)
    outlet1.apply_translation([-0.5, 0, 1.0])
    
    R_out2 = 0.3
    L_out2 = 2.0
    outlet2 = trimesh.creation.cylinder(radius=R_out2, height=L_out2, sections=32)
    outlet2.apply_translation([0, 0, L_out2/2])
    angle2 = np.radians(-30)
    rotation2 = trimesh.transformations.rotation_matrix(angle2, [1, 0, 0])
    outlet2.apply_transform(rotation2)
    outlet2.apply_translation([0.5, 0, 1.0])
    
    y_branch = trimesh.util.concatenate([inlet, outlet1, outlet2])
    
    return y_branch, R_in, L_in, R_out1, L_out1, R_out2, L_out2
