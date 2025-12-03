"""Tests for DesignSpec and design_from_spec."""

import pytest
import numpy as np
from vascular_lib.specs.design_spec import DesignSpec, EllipsoidSpec, InletSpec, ColonizationSpec, TreeSpec
from vascular_lib.api.design import design_from_spec


def test_ellipsoid_spec():
    spec = EllipsoidSpec(center=(0, 0, 0), semi_axes=(0.05, 0.04, 0.03))
    assert spec.type == "ellipsoid"
    d = spec.to_dict()
    spec2 = EllipsoidSpec.from_dict(d)
    assert spec2.center == spec.center


def test_design_from_spec():
    domain = EllipsoidSpec(center=(0, 0, 0), semi_axes=(0.05, 0.04, 0.03))
    inlet = InletSpec(position=(-0.04, 0, 0), radius=0.001, vessel_type="arterial")
    tissue_points = np.random.uniform(-0.03, 0.03, (30, 3))
    colonization = ColonizationSpec(
        tissue_points=tissue_points.tolist(),
        influence_radius=0.025,
        kill_radius=0.008,
        step_size=0.010,
        max_steps=10,
    )
    tree = TreeSpec(inlet=inlet, colonization=colonization)
    spec = DesignSpec(domain=domain, tree=tree)
    network = design_from_spec(spec)
    assert len(network.nodes) > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
