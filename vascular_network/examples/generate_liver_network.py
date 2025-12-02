#!/usr/bin/env python3
"""
Example script for generating liver vascular networks.

Usage:
    python examples/generate_liver_network.py --output output/liver_network --seed 42
"""

import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from generators.liver import LiverVascularConfig, generate_liver_vasculature
from generators.liver.export import export_to_python_module, export_to_json


def main():
    parser = argparse.ArgumentParser(
        description="Generate liver vascular network"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="output/liver_network",
        help="Output path (without extension)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility",
    )
    parser.add_argument(
        "--max-segments",
        type=int,
        default=5000,
        help="Maximum segments per tree",
    )
    parser.add_argument(
        "--arterial-first",
        action="store_true",
        default=True,
        help="Grow arterial tree first (default: True)",
    )
    
    args = parser.parse_args()
    
    config = LiverVascularConfig(random_seed=args.seed)
    config.growth.max_segments_per_tree = args.max_segments
    config.growth.arterial_first = args.arterial_first
    
    print("=" * 60)
    print("Liver Vascular Network Generator")
    print("=" * 60)
    print(f"Random seed: {config.random_seed}")
    print(f"Max segments per tree: {config.growth.max_segments_per_tree}")
    print(f"Arterial root radius: {config.murray.arterial_root_radius * 1000:.2f} mm")
    print(f"Venous root radius: {config.murray.venous_root_radius * 1000:.2f} mm")
    print(f"Min radius: {config.murray.min_radius * 1000:.2f} mm")
    print(f"Liver dimensions: {config.geometry.semi_axis_a * 100:.1f} x "
          f"{config.geometry.semi_axis_b * 100:.1f} x "
          f"{config.geometry.semi_axis_c * 100:.1f} cm")
    print()
    
    print("Generating vascular network...")
    arterial_tree, venous_tree = generate_liver_vasculature(config)
    
    print()
    print("Generation complete!")
    print(f"  Arterial tree: {len(arterial_tree.nodes)} nodes, "
          f"{len(arterial_tree.segments)} segments")
    print(f"  Venous tree: {len(venous_tree.nodes)} nodes, "
          f"{len(venous_tree.segments)} segments")
    print()
    
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    py_path = output_path.with_suffix(".py")
    json_path = output_path.with_suffix(".json")
    
    print("Exporting...")
    export_to_python_module(arterial_tree, venous_tree, config, py_path)
    export_to_json(arterial_tree, venous_tree, config, json_path)
    
    print()
    print("=" * 60)
    print("Done!")
    print(f"  Python module: {py_path}")
    print(f"  JSON file: {json_path}")
    print("=" * 60)


if __name__ == "__main__":
    main()
