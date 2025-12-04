# Migration Guide: Unit System Change (Meters to Millimeters)

## Overview

The vascular design library has been updated to use **millimeters** as the default unit instead of meters. This change affects all spatial parameters throughout the codebase.

**Key Change**: `1.0` now represents **1 millimeter** instead of 1 meter.

## What Changed

### Before (Meters)
```python
# Old code (meters)
domain = EllipsoidDomain(0.12, 0.10, 0.08)  # 120mm x 100mm x 80mm
inlet_position = (-0.10, 0.0, 0.0)  # -100mm
inlet_radius = 0.005  # 5mm
influence_radius = 0.010  # 10mm
```

### After (Millimeters)
```python
# New code (millimeters)
domain = EllipsoidDomain(120, 100, 80)  # 120mm x 100mm x 80mm
inlet_position = (-100, 0.0, 0.0)  # -100mm
inlet_radius = 5.0  # 5mm
influence_radius = 10.0  # 10mm
```

## Migration Steps

### 1. Update Spatial Parameters

Multiply all spatial values by **1000**:

- **Domain dimensions**: `0.12` → `120` (120mm)
- **Positions**: `(-0.10, 0.0, 0.0)` → `(-100, 0.0, 0.0)`
- **Radii**: `0.005` → `5.0` (5mm)
- **Distances**: `0.020` → `20.0` (20mm)
- **Clearances**: `0.001` → `1.0` (1mm)

### 2. Update Parameter Presets

If you're using custom colonization parameters:

```python
# Old (meters)
params = ColonizationSpec(
    influence_radius=0.015,  # 15mm
    kill_radius=0.003,       # 3mm
    step_size=0.001,         # 1mm
    min_radius=0.0003,       # 0.3mm
    min_clearance=0.0012,    # 1.2mm
)

# New (millimeters)
params = ColonizationSpec(
    influence_radius=15.0,   # 15mm
    kill_radius=3.0,         # 3mm
    step_size=1.0,           # 1mm
    min_radius=0.3,          # 0.3mm
    min_clearance=1.2,       # 1.2mm
)
```

### 3. Update Embedding Parameters

```python
# Old (meters)
result = embed_tree_as_negative_space(
    tree_stl_path='tree.stl',
    domain=domain,
    voxel_pitch=0.0005,      # 0.5mm
    shell_thickness=0.002,   # 2mm
)

# New (millimeters)
result = embed_tree_as_negative_space(
    tree_stl_path='tree.stl',
    domain=domain,
    voxel_pitch=0.5,         # 0.5mm
    shell_thickness=2.0,     # 2mm
    stl_units='auto',        # Auto-detect STL units
)
```

### 4. Update Validation Bounds

If you have custom validation logic, update bounds:

```python
# Old bounds (meters)
PARAM_BOUNDS = {
    "influence_radius": (0.001, 0.100, "m"),
    "kill_radius": (0.0001, 0.050, "m"),
}

# New bounds (millimeters)
PARAM_BOUNDS = {
    "influence_radius": (1.0, 100.0, "mm"),
    "kill_radius": (0.1, 50.0, "mm"),
}
```

## Physics Calculations

**Important**: Physics calculations (flow solver, Reynolds number) are automatically handled by the library. The flow solver internally converts geometry to SI units (meters) for correct physics, then converts results back to your input units.

You don't need to change anything in your flow analysis code:

```python
# This works the same way (units handled internally)
result = solve_flow(
    network,
    inlet_node_ids=[inlet_id],
    outlet_node_ids=[outlet_id],
    pin=13000.0,  # Pa
    pout=2000.0,  # Pa
)
```

## STL File Handling

The library now auto-detects STL file units:

```python
# Auto-detection (recommended)
result = embed_tree_as_negative_space(
    tree_stl_path='tree.stl',
    domain=domain,
    stl_units='auto',  # Auto-detect based on bounding box
)

# Explicit units (if auto-detection fails)
result = embed_tree_as_negative_space(
    tree_stl_path='tree.stl',
    domain=domain,
    stl_units='m',  # Force meters
)
```

## Common Issues

### Issue 1: Validation Errors

**Problem**: Getting validation errors like "influence_radius exceeds maximum"

**Solution**: You're still using meter-scale values. Multiply by 1000.

```python
# Wrong (still using meters)
params = get_preset("liver_arterial_dense")
params.influence_radius = 0.010  # ❌ This is now 0.01mm, too small!

# Correct (using millimeters)
params = get_preset("liver_arterial_dense")
params.influence_radius = 10.0  # ✓ This is 10mm
```

### Issue 2: Tiny Networks

**Problem**: Generated networks are extremely small

**Solution**: You forgot to multiply spatial parameters by 1000.

```python
# Wrong
domain = EllipsoidDomain(0.12, 0.10, 0.08)  # ❌ This is now 0.12mm x 0.10mm x 0.08mm!

# Correct
domain = EllipsoidDomain(120, 100, 80)  # ✓ This is 120mm x 100mm x 80mm
```

### Issue 3: Flow Solver Issues

**Problem**: Flow solver produces unrealistic results

**Solution**: Make sure you're passing the correct `geometry_units` parameter if your network uses non-standard units.

```python
# If your network is in meters (legacy)
result = solve_flow(
    network,
    geometry_units='m',  # Specify meters
)

# If your network is in millimeters (default)
result = solve_flow(
    network,
    geometry_units='mm',  # Default, can be omitted
)
```

## Backward Compatibility

The library maintains backward compatibility through:

1. **Auto-detection**: STL files are auto-detected based on bounding box size
2. **Explicit units**: You can specify `geometry_units` and `stl_units` parameters
3. **Validation warnings**: The validation system will warn if values seem incorrect

## Testing Your Migration

After migrating, verify your code:

1. **Check dimensions**: Print network bounds to verify they're in millimeters
2. **Run validation**: Use `validate_params()` to check parameter ranges
3. **Visual inspection**: Export to STL and check dimensions in a 3D viewer
4. **Flow analysis**: Verify Reynolds numbers are in expected ranges (< 2300 for laminar flow)

```python
# Verification example
network = design_from_spec(spec)

# Check bounds
bounds = network.domain.get_bounds()
print(f"Domain bounds: {bounds}")  # Should be in millimeters

# Check validation
params = spec.tree.colonization
is_valid, warnings = validate_params(params)
if not is_valid:
    print(f"Validation warnings: {warnings}")

# Check flow
result = solve_flow(network)
print(f"Max Reynolds: {result.metadata['max_reynolds']}")  # Should be < 2300
```

## Need Help?

If you encounter issues during migration:

1. Check that all spatial parameters are multiplied by 1000
2. Verify STL file units with `stl_units='auto'` or explicit specification
3. Review validation warnings for parameter range issues
4. Check the examples in `vascular_lib/examples/` for reference

## Summary

**Quick Checklist**:
- ✓ Multiply all spatial parameters by 1000
- ✓ Update domain dimensions
- ✓ Update positions and radii
- ✓ Update colonization parameters
- ✓ Update embedding parameters (voxel_pitch, shell_thickness)
- ✓ Add `stl_units='auto'` to embedding functions
- ✓ Run validation to check parameter ranges
- ✓ Test with small examples before migrating large codebases

The physics calculations are handled automatically - you only need to update the input values!
