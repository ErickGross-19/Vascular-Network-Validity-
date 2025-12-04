# Comprehensive Code Cleanup Plan

## Analysis Summary
- **Total Python files**: 99
- **Total functions**: 409
- **Duplicate function names**: 26
- **Critical cross-package duplicates**: 8

## Critical Duplicates Found

### Analysis Update (Phase 2)

After detailed analysis, the "duplicates" in generators/liver are **intentionally separate implementations**:

**Rationale:**
- generators/liver uses its own simple data structures (Node, Segment, VascularTree)
- vascular_lib uses more complex VascularNetwork with full graph topology
- generators is meant to be a standalone, simpler implementation for liver network generation
- Forcing generators to use vascular_lib would require complete rewrite of its data model

**Actual Duplicates to Address:**
- None in generators/liver (intentionally separate)
- Function name collisions exist but implementations serve different purposes
- No actual code duplication that needs cleanup

### Revised Priorities

**Priority 1: Code Organization** ✅
- Ensure clear separation between vascular_lib (general framework) and generators (specific implementations)
- Document the relationship in READMEs
- Verify no unintended dependencies

**Priority 2: Test Fixes**
- Fix test_component_flows.py::test_compute_component_flows_dual_tree (bifurcate API mismatch)
- Ensure all tests pass

**Priority 3: Documentation**
- Update READMEs to clarify package purposes
- Document that generators is intentionally separate from vascular_lib

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

### Phase 2: Analysis & Documentation ✅ COMPLETE
- [x] Analyzed generators/liver implementation
- [x] Determined it's intentionally separate from vascular_lib
- [x] Updated cleanup plan to reflect reality
- [x] No deduplication needed - different purposes

### Phase 3: Fix Test Failures
- [ ] Fix test_component_flows.py bifurcate() API mismatch
- [ ] Ensure all vascular_lib tests pass
- [ ] Run vascular_network tests

### Phase 4: Documentation Updates
- [ ] Update vascular_lib/README.md to clarify purpose
- [ ] Update generators/README.md to explain it's standalone
- [ ] Update root README.md to explain package relationships

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
