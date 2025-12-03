# Vascular Library LLM Enhancement Roadmap

This document outlines the comprehensive improvements to make vascular_lib more suitable for LLM-driven vascular network design.

## Overview

**Total Estimated Effort**: 20-40 hours
**Implementation Strategy**: Systematic, one feature at a time
**Priority**: High-impact LLM API features first, then quality improvements, then advanced features

---

## Phase 1: Core LLM API (Priority: CRITICAL)
**Estimated Time**: 6-9 hours
**Goal**: Provide stable, high-level API for LLM agents

### A1. DesignSpec Dataclasses ✅ NEXT
- [ ] Create `DomainSpec` (base class)
- [ ] Create `EllipsoidSpec` and `BoxSpec`
- [ ] Create `InletSpec` and `OutletSpec`
- [ ] Create `ColonizationSpec` with all space colonization parameters
- [ ] Create `TreeSpec` for single-tree designs
- [ ] Create `DualTreeSpec` for arterial+venous designs
- [ ] Create `DesignSpec` as top-level container
- [ ] Add `to_dict()` and `from_dict()` methods for JSON serialization
- [ ] Add JSON schema documentation
- **Files**: `vascular_lib/specs/design_spec.py`

### A2. design_from_spec() Function
- [ ] Implement `design_from_spec(spec: DesignSpec) -> VascularNetwork`
- [ ] Handle single-tree and dual-tree specs
- [ ] Create domain from spec
- [ ] Add inlets and outlets
- [ ] Run space colonization with specified parameters
- [ ] Return complete network
- **Files**: `vascular_lib/api/design.py`

### A3. evaluate_network() Function
- [ ] Create `EvalMetrics` dataclass with:
  - Coverage metrics (coverage_fraction, unperfused_points, uniformity)
  - Flow metrics (flow_balance_error, turbulent_fraction, pressure_stats)
  - Structure metrics (total_length, degree_histogram, branching_angles, Murray_deviation)
  - Validity metrics (watertight, self_intersections, param_warnings)
- [ ] Create `EvalScores` dataclass with:
  - coverage_score, flow_score, structure_score, overall_score
- [ ] Create `EvalResult` dataclass combining metrics and scores
- [ ] Implement `evaluate_network(network, tissue_points, config) -> EvalResult`
- [ ] Add configurable scoring weights
- **Files**: `vascular_lib/specs/eval_result.py`, `vascular_lib/api/evaluate.py`

---

## Phase 2: Space Colonization Quality (Priority: HIGH)
**Estimated Time**: 4-6 hours
**Goal**: Improve generated network quality

### B1. Wire Up Bifurcation Controls ✅ DONE (mostly)
- [x] Bifurcation parameters already added
- [ ] Verify bifurcation logic works correctly
- [ ] Add tests for bifurcation triggering
- **Files**: `vascular_lib/ops/space_colonization.py`

### B2. Max Curvature Constraints
- [ ] Add `max_curvature_deg` parameter to `SpaceColonizationParams`
- [ ] In `space_colonization_step`, check angle between new direction and parent direction
- [ ] If angle > max_curvature_deg, project onto cone around parent direction
- [ ] Add tests for curvature limiting
- **Files**: `vascular_lib/ops/space_colonization.py`

### B3. Self-Avoidance / Clearance Checks
- [ ] Add `min_clearance` parameter to `BranchingConstraints`
- [ ] In `grow_branch`, check distance to all existing segments
- [ ] If clearance < min_clearance, try jittering direction or skip growth
- [ ] Add clearance violation tracking to `OperationResult`
- [ ] Add tests for self-avoidance
- **Files**: `vascular_lib/ops/growth.py`, `vascular_lib/ops/collision.py`

### B4. Relaxed Murray's Law at Junctions
- [ ] Add `enforce_murray` parameter to `BranchingConstraints`
- [ ] After bifurcation, check `r_parent^3 ≈ r_child1^3 + r_child2^3`
- [ ] If violated and enforce_murray=True, rescale child radii
- [ ] Add Murray deviation to evaluation metrics
- [ ] Add tests for Murray's law enforcement
- **Files**: `vascular_lib/ops/growth.py`

### B5. Parameter Presets
- [ ] Create `vascular_lib/rules/presets.py`
- [ ] Add `SpaceColonizationParams.liver_arterial()`
- [ ] Add `SpaceColonizationParams.liver_venous()`
- [ ] Add `SpaceColonizationParams.sparse_debug()`
- [ ] Add `SpaceColonizationParams.dense_production()`
- [ ] Document all presets
- **Files**: `vascular_lib/rules/presets.py`

### B6. Parameter Validation
- [ ] Create `PARAM_BOUNDS` dict with min/max for each parameter
- [ ] Implement `validate_params(params) -> List[str]` returning warnings
- [ ] Check: step_size > 0, min_radius > 0, kill_radius < influence_radius, etc.
- [ ] Add tests for validation
- **Files**: `vascular_lib/rules/validation.py`

---

## Phase 3: Flow & Physics (Priority: MEDIUM)
**Estimated Time**: 3-4 hours
**Goal**: Standardize flow analysis and add quality scoring

### C1. Standardize Flow Solver Usage
- [ ] Ensure `solve_flow()` and `compute_component_flows()` are used everywhere
- [ ] Update `evaluate_network()` to use standardized flow API
- [ ] Deprecate old `estimate_flows()` paths (keep for backward compat)
- [ ] Add tests for flow solver
- **Files**: `vascular_lib/analysis/solver.py`, `vascular_lib/api/evaluate.py`

### C2. Flow Quality Score
- [ ] Implement `compute_flow_score(flow_result) -> float`
- [ ] Combine: flow_balance_error, turbulent_fraction, pressure_uniformity
- [ ] Formula: `1.0 - (balance_error + 0.1*turbulent_fraction)`
- [ ] Add to `EvalScores`
- [ ] Add tests
- **Files**: `vascular_lib/api/evaluate.py`

---

## Phase 4: Mesh & Voxel Robustness (Priority: MEDIUM)
**Estimated Time**: 3-4 hours
**Goal**: Make voxelization more reliable and user-friendly

### D1. Fix Voxel Pitch Handling
- [ ] Remove contradictory min_pitch clamp
- [ ] Compute recommended pitch from bounding box
- [ ] Only increase pitch if necessary (not decrease below user request)
- [ ] Add clear error messages with recommended pitch
- [ ] Add tests
- **Files**: `vascular_network/mesh/repair.py`

### D2. Auto-Pitch Selection
- [ ] Implement `auto_voxel_pitch(mesh, target_resolution=256) -> float`
- [ ] Compute: `pitch = max_extent / target_resolution`
- [ ] Add to `validate_and_repair_geometry()` as default
- [ ] Add tests
- **Files**: `vascular_network/mesh/repair.py`

### D3. Preview vs Production Modes
- [ ] Add `mode` parameter to `validate_and_repair_geometry()`
- [ ] Preview: coarse pitch (128 voxels), light morphology (1 iter each)
- [ ] Production: fine pitch (512 voxels), full morphology (default iters)
- [ ] Add tests
- **Files**: `vascular_network/pipeline.py`

### D4. Configurable Node Sphere Scaling
- [ ] Add `node_sphere_scale` parameter to `export_stl()`
- [ ] Default to current behavior (backward compat)
- [ ] Allow user to make nodes larger/smaller
- [ ] Add tests
- **Files**: `vascular_lib/adapters/mesh_export.py`

### D5. Robust Error Types
- [ ] Create `vascular_lib/core/errors.py` with error codes
- [ ] Define: `PITCH_TOO_COARSE`, `MORPH_REMOVED_ALL_VOXELS`, `GEOMETRY_TOO_THIN`
- [ ] Return error codes in `OperationResult.metadata["error_code"]`
- [ ] Add to documentation
- **Files**: `vascular_lib/core/errors.py`, `vascular_network/mesh/repair.py`

---

## Phase 5: LLM Agent Support (Priority: HIGH)
**Estimated Time**: 4-5 hours
**Goal**: Complete end-to-end workflow for LLM agents

### E1. run_experiment() Helper
- [ ] Implement `run_experiment(spec: DesignSpec, output_dir: Path) -> EvalResult`
- [ ] Build network from spec
- [ ] Run evaluation
- [ ] Export STL
- [ ] Save spec.json, network.json, eval.json
- [ ] Return EvalResult with file paths
- [ ] Add tests
- **Files**: `vascular_lib/api/experiment.py`

### E2. Experiment Logging
- [ ] Create run directory: `runs/YYYY-MM-DD_HHMMSS/`
- [ ] Save: spec.json, network.json, eval.json, network.stl
- [ ] Add experiment metadata (timestamp, git commit, parameters)
- [ ] Add `list_experiments()` to query past runs
- [ ] Add tests
- **Files**: `vascular_lib/api/experiment.py`

### E3. Progress Callbacks
- [ ] Add `progress_callback` parameter to `space_colonization_step()`
- [ ] Add `progress_callback` parameter to `export_stl()`
- [ ] Add `progress_callback` parameter to `validate_and_repair_geometry()`
- [ ] Callback signature: `(current: int, total: int, message: str) -> None`
- [ ] Add tests
- **Files**: Multiple

### E4. Agent Recipe Documentation
- [ ] Create `docs/llm_agent_guide.md`
- [ ] Document: design_from_spec usage
- [ ] Document: evaluate_network usage
- [ ] Document: iterative improvement loop
- [ ] Add example agent code
- **Files**: `docs/llm_agent_guide.md`

---

## Phase 6: Visualization & Debugging (Priority: LOW)
**Estimated Time**: 3-4 hours
**Goal**: Better debugging and inspection tools

### F1. Improved Visualizations
- [ ] Implement `plot_network(network, color_by="vessel_type")`
- [ ] Support color_by: "vessel_type", "flow_rate", "reynolds", "radius"
- [ ] Add 3D centerline plots with radii
- [ ] Add tests
- **Files**: `vascular_lib/visualization/network_plots.py`

### F2. Sparse Demo Preset
- [ ] Add `SpaceColonizationParams.sparse_debug()`
- [ ] Very thin branches, large domain, few steps
- [ ] Good for quick visual inspection
- [ ] Add example
- **Files**: `vascular_lib/rules/presets.py`

### F3. Branch Statistics
- [ ] Implement `compute_branch_stats(network) -> dict`
- [ ] Return: degree_histogram, avg_path_length, branching_angle_distribution
- [ ] Add to evaluation metrics
- [ ] Add tests
- **Files**: `vascular_lib/analysis/structure.py`

---

## Phase 7: Advanced Features (Priority: FUTURE)
**Estimated Time**: 8-12 hours
**Goal**: Advanced editing and optimization

### B7. Edit Operations (FUTURE)
- [ ] Implement `prune_subtree(network, root_node_id)`
- [ ] Implement `reroute_branch(network, node_id, new_direction)`
- [ ] Implement `local_colonization_step(network, region_bbox, params)`
- [ ] Implement `connect_nodes(network, node_id1, node_id2)` (anastomosis variant)
- [ ] Add tests
- **Files**: `vascular_lib/ops/editing.py`

### C3. Shear-Stress Remodeling (FUTURE)
- [ ] Implement simple remodeling: thicken high-Re segments, thin low-Re
- [ ] Add `remodel_by_shear_stress(network, flow_result, iterations=10)`
- [ ] Add tests
- **Files**: `vascular_lib/analysis/remodeling.py`

---

## Implementation Order

1. **A1**: DesignSpec dataclasses (foundation)
2. **A2**: design_from_spec() (core API)
3. **A3**: evaluate_network() (core API)
4. **B2**: Max curvature constraints (quality)
5. **B3**: Self-avoidance checks (quality)
6. **B5**: Parameter presets (usability)
7. **B6**: Parameter validation (robustness)
8. **E1**: run_experiment() (end-to-end)
9. **C1**: Standardize flow solver (cleanup)
10. **C2**: Flow quality score (evaluation)
11. **D1-D2**: Voxel pitch improvements (robustness)
12. **E2**: Experiment logging (tracking)
13. **F1-F3**: Visualizations (debugging)
14. **B4**: Murray's law enforcement (advanced)
15. **D3-D5**: Advanced voxel features (polish)
16. **E3-E4**: Progress callbacks & docs (polish)

---

## Success Criteria

- [ ] LLM can create network from JSON spec
- [ ] LLM can evaluate network and get structured feedback
- [ ] LLM can run complete experiment with one function call
- [ ] Generated networks have smooth curvature (no sharp kinks)
- [ ] Generated networks avoid self-collisions
- [ ] Parameter validation prevents invalid configurations
- [ ] Voxelization is robust and provides clear error messages
- [ ] All features have unit tests
- [ ] Documentation is complete and clear

---

## Testing Strategy

- Unit tests for each new function
- Integration tests for end-to-end workflows
- Regression tests to ensure backward compatibility
- Performance tests for large networks (1000+ nodes)
- Determinism tests (same seed → same result)

---

## Documentation Updates

- [ ] Update vascular_lib/README.md with new APIs
- [ ] Create docs/llm_agent_guide.md
- [ ] Add docstrings to all new functions
- [ ] Add examples to vascular_lib/examples/
- [ ] Update CHANGELOG.md

---

## Notes

- **Backward Compatibility**: All new features are opt-in with safe defaults
- **Determinism**: All random operations use network.id_gen.rng
- **Performance**: Angle spread computation capped at 50 attractions per node
- **Dependencies**: Only tqdm added (already in requirements.txt)
- **Error Handling**: Structured error codes for machine-readable feedback
