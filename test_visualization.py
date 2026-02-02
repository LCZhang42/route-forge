"""
Quick test script to demonstrate path visualization on moonboard.
Creates a sample path and visualizes it.
"""

import json
from pathlib import Path
from visualize_path import MoonBoardVisualizer

# Create sample climbing paths
sample_paths = {
    "config": {
        "grade": "7A",
        "temperature": 1.0,
        "min_holds": 3,
        "max_holds": 30,
        "constraints_enabled": True
    },
    "paths": [
        {
            "grade": "7A",
            "holds": [[5, 4], [6, 6], [7, 8], [6, 10], [7, 12], [6, 15], [6, 17]],
            "num_holds": 7
        },
        {
            "grade": "7A",
            "holds": [[3, 3], [4, 5], [5, 7], [6, 9], [7, 11], [8, 13], [9, 15], [9, 17]],
            "num_holds": 8
        },
        {
            "grade": "7A",
            "holds": [[2, 2], [3, 4], [4, 6], [5, 8], [6, 10], [7, 12], [8, 14], [9, 16], [10, 18]],
            "num_holds": 9
        }
    ]
}

# Save sample paths to JSON
output_json = Path("test_paths_sample.json")
with open(output_json, 'w') as f:
    json.dump(sample_paths, f, indent=2)

print(f"Created sample paths file: {output_json}")

# Visualize the paths
print("\nVisualizing paths on moonboard...")

try:
    visualizer = MoonBoardVisualizer("data/moonboard2016Background.jpg")
    
    output_dir = Path("test_visualizations")
    output_dir.mkdir(exist_ok=True)
    
    for idx, path_data in enumerate(sample_paths["paths"]):
        holds = path_data["holds"]
        grade = path_data["grade"]
        
        output_path = output_dir / f"test_path_{idx+1}_{grade}.png"
        visualizer.draw_path(holds, grade, str(output_path))
        print(f"  [OK] Created: {output_path}")
    
    print(f"\n[SUCCESS] Check the '{output_dir}' folder for visualizations.")
    print(f"\nGenerated {len(sample_paths['paths'])} visualization(s):")
    for idx, path_data in enumerate(sample_paths["paths"]):
        print(f"  - Path {idx+1}: {path_data['num_holds']} holds, Grade {path_data['grade']}")

except Exception as e:
    print(f"\n[ERROR] {e}")
    print("\nMake sure:")
    print("  1. OpenCV is installed: pip install opencv-python")
    print("  2. Moonboard image exists at: data/moonboard2016Background.jpg")
