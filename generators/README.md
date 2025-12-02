# Vascular Network Generators

This package contains tools for generating synthetic vascular networks with realistic topology and geometry.

## Liver Vascular Network Generator

The liver vascular network generator creates realistic arterial and venous trees inside a liver-shaped domain, following biophysical rules like Murray's law and space-filling constraints.

### Features

- **Biophysical accuracy**: Implements Murray's law for radius scaling at bifurcations
- **Space-filling growth**: Trees grow to fill the liver volume efficiently
- **Collision avoidance**: Arterial and venous trees approach each other without intersecting
- **Configurable parameters**: Control all aspects of generation (radii, branching, growth)
- **Multiple output formats**: Export to Python modules and JSON files

### Quick Start

```python
from generators.liver import LiverVascularConfig, generate_liver_vasculature

# Use default configuration
arterial_tree, venous_tree = generate_liver_vasculature()

# Or customize parameters
config = LiverVascularConfig(random_seed=42)
config.murray.arterial_root_radius = 0.006  # 6 mm
config.growth.max_segments_per_tree = 1000
arterial_tree, venous_tree = generate_liver_vasculature(config)
```

### Command-Line Usage

```bash
# Generate with default parameters
python examples/generate_liver_network.py --output output/liver_network

# Customize generation
python examples/generate_liver_network.py \
    --output output/my_liver \
    --seed 123 \
    --max-segments 2000
```

### Output Format

The generator produces two files:

1. **Python module** (`.py`): Contains `NODES`, `SEGMENTS`, `ROOTS`, and `CONFIG` variables
2. **JSON file** (`.json`): Same data in JSON format for non-Python consumers

Example usage of generated module:

```python
import sys
sys.path.insert(0, 'output')
import liver_network

# Access nodes and segments
arterial_nodes = liver_network.get_nodes_by_tree('arterial')
venous_segments = liver_network.get_segments_by_tree('venous')

# Get root IDs
arterial_root = liver_network.ROOTS['arterial']
venous_root = liver_network.ROOTS['venous']
```

### Configuration

The generator is highly configurable through the `LiverVascularConfig` dataclass:

#### Geometry Configuration
- Liver ellipsoid dimensions (semi-axes)
- Root positions for arterial and venous trees
- Meeting shell parameters (where trees approach)

#### Murray's Law Configuration
- Gamma exponent (default: 3.0)
- Root radii for arterial and venous trees
- Minimum radius (capillary scale)
- Bifurcation split ratios

#### Branching Configuration
- Maximum branch order
- Branching probability parameters
- Branch angle distributions
- Maximum curvature per step
- Step size factor

#### Growth Configuration
- Growth mode (arterial-first or interleaved)
- Maximum segments per tree
- Collision margins
- Boundary avoidance distances
- Direction bias weights

### Algorithm Overview

The generator uses an arterial-first growth strategy:

1. **Initialize**: Create root nodes for arterial and venous trees at opposite sides of liver
2. **Grow arterial tree**: Extend tips incrementally with branching and collision avoidance
3. **Grow venous tree**: Same process but with collision avoidance against arterial tree
4. **Export**: Save trees to Python module and JSON file

Growth process for each tree:
- Maintain active tips (growth fronts)
- For each tip: decide whether to continue or branch based on radius and probability
- Compute new positions with biased directions (towards center, away from boundary)
- Check validity (inside liver, no collisions, within meeting shell)
- Add new nodes and segments, update active tips
- Terminate when max segments reached or all tips exhausted

### Biophysical Rules

**Murray's Law**: At each bifurcation, child radii satisfy:
```
r_parent^3 = r_child1^3 + r_child2^3
```

**Branching Angles**: Sampled from normal distribution (mean ~50°, std ~15°)

**Radius Tapering**: Vessels get progressively smaller from root to leaves

**Collision Detection**: Uses spatial grid indexing for efficient neighbor queries

### Future Enhancements

- Interleaved growth mode (alternate between arterial and venous)
- Space colonization algorithm for better coverage
- STL export with cylindrical sweep
- Integration with existing validation pipeline
- More realistic liver geometry (from medical imaging)
- Lobar structure (separate left/right lobes)

### References

- Murray, C. D. (1926). "The Physiological Principle of Minimum Work"
- Schreiner, W. & Buxbaum, P. F. (1993). "Computer-optimization of vascular trees"
- Karch, R. et al. (1999). "A three-dimensional model for arterial tree representation"
