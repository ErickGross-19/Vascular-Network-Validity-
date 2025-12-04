"""Tests for parameter presets and validation."""

import pytest
from vascular_lib.params import get_preset, list_presets, validate_params


def test_list_presets():
    presets = list_presets()
    assert isinstance(presets, list)
    assert "liver_arterial_dense" in presets
    assert "sparse_debug" in presets


def test_get_preset():
    params = get_preset("liver_arterial_dense")
    assert params.influence_radius == 10.0  # 10mm (updated from 0.010m)
    assert params.max_curvature_deg == 60.0


def test_validate_params():
    params = get_preset("sparse_debug")
    is_valid, warnings = validate_params(params)
    assert is_valid is True


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
