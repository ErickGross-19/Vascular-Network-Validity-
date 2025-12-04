# Comprehensive Code Cleanup Plan

## Analysis Summary
- **Total Python files**: 99
- **Total functions**: 409
- **Duplicate function names**: 26
- **Critical cross-package duplicates**: 8

## Critical Duplicates Found

### Priority 1: Spatial Indexing (generators/liver/tree.py → vascular_lib/spatial/)
- `_get_cell_coords` (2 occurrences)
- `_get_cells_for_segment` (2 occurrences)
- `query_nearby_segments` (2 occurrences)
- `_point_to_segment_distance` (2 occurrences)

### Priority 2: Network Methods (generators/liver/tree.py → vascular_lib/core/)
- `add_node` (3 occurrences)
- `add_segment` (3 occurrences)
- `get_node` (2 occurrences)

### Priority 3: Domain Methods (generators/liver/geometry.py → vascular_lib/core/)
- `distance_to_boundary` (5 occurrences - 4 in vascular_lib/core/domain.py as base+subclasses, 1 in generators)

## Cleanup Phases

### Phase 0: Safety & Preparation ✅ COMPLETE
- [x] Add CI workflow
- [x] Generate function inventory
- [x] Identify critical duplicates

### Phase 1: Fix API Mismatches (IN PROGRESS)
- [ ] Review design_spec_example.py API usage
- [ ] Decide: Add convenience constructors OR update examples to match actual API
- [ ] Fix all examples to use correct API
- [ ] Test examples work

### Phase 2: Deduplicate Spatial Indexing
- [ ] Review generators/liver/tree.py spatial methods
- [ ] Make generators import from vascular_lib.spatial.grid_index
- [ ] Add deprecation wrappers if needed
- [ ] Test generators still work

### Phase 3: Deduplicate Network Methods
- [ ] Make generators use vascular_lib.core.network
- [ ] Update generators to use VascularNetwork data model
- [ ] Test generators still work

### Phase 4: Deduplicate Domain Methods
- [ ] Review generators/liver/geometry.py
- [ ] Use vascular_lib.core.domain classes or create LobedLiverDomain subclass
- [ ] Test generators still work

### Phase 5: Code Organization
- [ ] Review folder structure
- [ ] Move any misplaced code to canonical locations
- [ ] Update __init__.py exports
- [ ] Remove any remaining dead code

### Phase 6: Final Verification
- [ ] Run all tests
- [ ] Run all examples
- [ ] Update documentation
- [ ] Create PR

## Notes
- Keep backward compatibility with deprecation warnings
- Test after each phase
- Commit incrementally
