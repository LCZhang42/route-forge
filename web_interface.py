"""
Gradio Web Interface for MoonBoard Climb Path Generator.

Provides an interactive UI for generating and visualizing climbing paths.
"""

import gradio as gr
import torch
import sys
import json
from pathlib import Path
from typing import Tuple, Optional
import numpy as np
from PIL import Image
import cv2

sys.path.append(str(Path(__file__).parent))

from backend.models.tokenizer import ClimbPathTokenizer
from backend.training.generate_paths import load_model, generate_climb_path
from backend.models.valid_holds import load_valid_holds_from_dataset
from backend.models.climb_constraints import load_constraints, get_random_hold_count
from visualize_path import MoonBoardVisualizer
import ast
import pandas as pd


# Global variables for model and resources
model = None
tokenizer = None
device = None
valid_holds = None
visualizer = None
grade_constraints = None
endpoints_by_grade = None


def load_benchmark_endpoints(csv_path='data/moonboard_test_benchmark.csv'):
    """
    Load start and end points from benchmark dataset.
    
    Returns:
        dict: {grade: {'start_points': [...], 'end_points': [...]}}
    """
    df = pd.read_csv(csv_path)
    
    # Parse start_holds and end_holds if they're strings
    if isinstance(df['start_holds'].iloc[0], str):
        df['start_holds'] = df['start_holds'].apply(ast.literal_eval)
    if isinstance(df['end_holds'].iloc[0], str):
        df['end_holds'] = df['end_holds'].apply(ast.literal_eval)
    
    # Group by grade
    endpoints_by_grade = {}
    for grade in df['grade'].unique():
        grade_df = df[df['grade'] == grade]
        
        # Collect all start points (flatten list of lists)
        start_points = []
        for holds_list in grade_df['start_holds']:
            start_points.extend(holds_list)
        
        # Collect all end points (flatten list of lists)
        end_points = []
        for holds_list in grade_df['end_holds']:
            end_points.extend(holds_list)
        
        endpoints_by_grade[grade] = {
            'start_points': start_points,
            'end_points': end_points
        }
        
        print(f"  Grade {grade}: {len(start_points)} start points, {len(end_points)} end points")
    
    return endpoints_by_grade


def initialize_resources():
    """Initialize model, tokenizer, and valid holds."""
    global model, tokenizer, device, valid_holds, visualizer, endpoints_by_grade
    
    print("Initializing resources...")
    
    # Load model
    checkpoint_path = Path("checkpoints/climb_path_cpu/latest.pt")
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Model checkpoint not found at {checkpoint_path}")
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading model on {device}...")
    model, config = load_model(str(checkpoint_path), device)
    tokenizer = ClimbPathTokenizer()
    
    # Load valid holds
    train_csv_path = Path("data/moonboard_train_quality.csv")
    if train_csv_path.exists():
        print("Loading valid MoonBoard 2016 hold positions...")
        valid_holds = load_valid_holds_from_dataset(str(train_csv_path))
        print(f"Loaded {len(valid_holds)} valid holds")
    else:
        print(f"Warning: Training data not found at {train_csv_path}")
        valid_holds = None
    
    # Initialize visualizer
    background_path = Path("data/moonboard2016Background.jpg")
    if not background_path.exists():
        raise FileNotFoundError(f"Background image not found at {background_path}")
    visualizer = MoonBoardVisualizer(str(background_path))
    
    # Load grade constraints
    constraints_path = Path("data/grade_constraints.json")
    if constraints_path.exists():
        print("Loading grade constraints...")
        global grade_constraints
        grade_constraints = load_constraints(str(constraints_path))
        print(f"Loaded constraints for {len(grade_constraints)} grades")
    else:
        print(f"Warning: Grade constraints not found at {constraints_path}")
        grade_constraints = None
    
    # Load benchmark endpoints
    benchmark_path = Path("data/moonboard_test_benchmark.csv")
    if benchmark_path.exists():
        print("Loading benchmark endpoints...")
        endpoints_by_grade = load_benchmark_endpoints(str(benchmark_path))
        print(f"Loaded endpoints for {len(endpoints_by_grade)} grades")
    else:
        print(f"Warning: Benchmark data not found at {benchmark_path}")
        endpoints_by_grade = None
    
    print("Resources initialized successfully!")


def generate_and_visualize(
    grade: str,
    valid_holds_only: bool,
    temperature: float,
    use_endpoint_conditioning: bool
) -> Tuple[Optional[Image.Image], str]:
    """
    Generate a climb path and create visualization.
    
    Hold count is automatically determined based on grade distribution.
    
    Args:
        grade: Climbing grade
        valid_holds_only: Whether to only generate valid holds
        temperature: Sampling temperature
        use_endpoint_conditioning: Whether to use random start/end from benchmark
        
    Returns:
        (image, info_text) tuple
    """
    try:
        # Check if using endpoint conditioning
        if use_endpoint_conditioning:
            if endpoints_by_grade is None or grade not in endpoints_by_grade:
                return None, f"**Error:** Endpoint conditioning enabled but no benchmark data available for grade {grade}"
            
            # Randomly select start and end points from benchmark
            start_points = endpoints_by_grade[grade]['start_points']
            end_points = endpoints_by_grade[grade]['end_points']
            
            start_hold = tuple(start_points[np.random.randint(len(start_points))])
            end_hold = tuple(end_points[np.random.randint(len(end_points))])
            
            print(f"Generating path with endpoint conditioning for grade {grade}")
            print(f"  Start: {start_hold}, End: {end_hold}")
            
            # Encode grade
            grade_token = tokenizer.encode_grade(grade)
            
            # Generate path with endpoints
            generated_tokens = model.generate(
                grade_token=grade_token,
                tokenizer=tokenizer,
                start_hold=start_hold,
                end_hold=end_hold,
                max_length=64,
                temperature=temperature,
                device=device,
            )
            
            # Debug: print generated tokens
            print(f"  Generated tokens: {generated_tokens.tolist()[:10]}...")
            print(f"  Token at position 1 (grade): {generated_tokens[1].item()}")
            
            # Decode
            generated_grade, start_dec, end_dec, intermediate_holds = tokenizer.decode_with_endpoints(generated_tokens)
            holds = [start_dec] + list(intermediate_holds) + [end_dec]
            
        else:
            # Original generation without endpoint conditioning
            # Determine whether to use valid holds constraint
            holds_constraint = valid_holds if valid_holds_only else None
            
            # Get random hold count based on grade distribution
            if grade_constraints and grade in grade_constraints:
                min_holds = max(3, int(grade_constraints[grade]['hold_count_mean'] - grade_constraints[grade]['hold_count_std']))
                max_holds = min(30, int(grade_constraints[grade]['hold_count_mean'] + grade_constraints[grade]['hold_count_std']))
            else:
                min_holds = 5
                max_holds = 15
            
            # Generate path
            print(f"Generating path for grade {grade} ({min_holds}-{max_holds} holds)...")
            generated_grade, holds = generate_climb_path(
                model=model,
                tokenizer=tokenizer,
                grade=grade,
                device=device,
                temperature=temperature,
                min_holds=min_holds,
                max_holds=max_holds,
                use_constraints=True,
                valid_holds=holds_constraint,
                use_reachability=False,
            )
        
        # Create visualization
        import tempfile
        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp:
            output_path = tmp.name
        
        visualizer.draw_path(holds, generated_grade, output_path)
        
        # Load image
        img = Image.open(output_path)
        
        # Create info text
        info_lines = [
            f"**Grade:** {generated_grade}",
            f"**Number of Holds:** {len(holds)}",
            f"**Mode:** {'Endpoint Conditioning' if use_endpoint_conditioning else 'Free Generation'}",
            f"**Temperature:** {temperature}",
        ]
        
        if use_endpoint_conditioning:
            info_lines.append(f"**Start Hold:** [{holds[0][0]}, {holds[0][1]}] = {chr(ord('A') + holds[0][0])}{holds[0][1]}")
            info_lines.append(f"**End Hold:** [{holds[-1][0]}, {holds[-1][1]}] = {chr(ord('A') + holds[-1][0])}{holds[-1][1]}")
            info_lines.append(f"**Intermediate Holds:** {len(holds) - 2}")
        else:
            info_lines.append(f"**Valid Holds Only:** {'Yes' if valid_holds_only else 'No'}")
        
        info_lines.extend(["", "**Hold Sequence:**"])
        
        for i, (x, y) in enumerate(holds[:10]):  # Show first 10 holds
            col = chr(ord('A') + x)
            info_lines.append(f"{i+1}. {col}{y} → [{x}, {y}]")
        
        if len(holds) > 10:
            info_lines.append(f"... and {len(holds) - 10} more holds")
        
        info_text = "\n".join(info_lines)
        
        return img, info_text
        
    except Exception as e:
        error_msg = f"**Error:** {str(e)}\n\nPlease check the console for details."
        print(f"Error during generation: {e}")
        import traceback
        traceback.print_exc()
        return None, error_msg


def create_interface():
    """Create and configure the Gradio interface."""
    
    # Initialize resources
    try:
        initialize_resources()
    except Exception as e:
        print(f"Failed to initialize resources: {e}")
        raise
    
    # Get available grades
    grades = tokenizer.GRADES
    
    # Create interface
    with gr.Blocks(title="MoonBoard Climb Path Generator") as interface:
        gr.Markdown(
            """
            # 🧗 MoonBoard Climb Path Generator
            
            Generate realistic climbing paths for the MoonBoard 2016 using AI.
            Select your desired grade, configure generation parameters, and visualize the path!
            """
        )
        
        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown("### Generation Settings")
                
                grade_dropdown = gr.Dropdown(
                    choices=grades,
                    value="7A",
                    label="Climbing Grade",
                    info="Select the difficulty grade for the climb"
                )
                
                endpoint_conditioning_toggle = gr.Checkbox(
                    value=True,
                    label="Endpoint Conditioning",
                    info="Use random start/end from benchmark dataset (recommended for trained model)"
                )
                
                valid_holds_toggle = gr.Checkbox(
                    value=False,
                    label="Valid Holds Only",
                    info="Only generate holds that exist on MoonBoard 2016 (only for free generation)"
                )
                
                gr.Markdown("### Advanced Parameters")
                
                temperature_slider = gr.Slider(
                    minimum=0.5,
                    maximum=2.0,
                    value=1.0,
                    step=0.1,
                    label="Temperature",
                    info="Higher = more random/creative paths"
                )
                
                gr.Markdown(
                    """
                    **Note:** Hold count is automatically determined based on the 
                    grade's distribution in the training dataset for realistic paths.
                    """
                )
                
                generate_btn = gr.Button("🎯 Generate Path", variant="primary", size="lg")
                
                gr.Markdown(
                    """
                    ---
                    **Legend:**
                    - 🟢 Green circle = Start hold
                    - 🔵 Blue circles = Middle holds
                    - 🔴 Red circle = End hold
                    """
                )
            
            with gr.Column(scale=2):
                gr.Markdown("### Generated Path Visualization")
                
                output_image = gr.Image(
                    label="MoonBoard Path",
                    type="pil",
                    height=600
                )
                
                info_text = gr.Markdown(
                    label="Path Information",
                    value="Click **Generate Path** to create a climbing route!"
                )
        
        # Connect the generate button
        generate_btn.click(
            fn=generate_and_visualize,
            inputs=[
                grade_dropdown,
                valid_holds_toggle,
                temperature_slider,
                endpoint_conditioning_toggle
            ],
            outputs=[output_image, info_text]
        )
        
        gr.Markdown(
            """
            ---
            ### About
            
            This tool uses a transformer-based AI model trained on thousands of real MoonBoard climbs.
            The model learns patterns in climbing sequences and generates new, realistic paths.
            
            **Endpoint Conditioning Mode (Recommended):** The model generates intermediate holds between 
            randomly selected start and end points from the benchmark dataset. This produces more realistic 
            and controlled paths since the model was trained with this approach.
            
            **Free Generation Mode:** The model generates the entire path from scratch without constraints.
            Use this for more creative exploration (requires model trained without endpoint conditioning).
            
            **Valid Holds Mode:** Only applies to free generation. Restricts holds to positions where 
            physical boulders exist on the MoonBoard 2016.
            
            **Tips:**
            - Lower temperature (0.5-0.8) = More conservative, realistic paths
            - Higher temperature (1.2-2.0) = More creative, varied paths
            - Endpoint conditioning is recommended for the current trained model
            """
        )
    
    return interface


def main():
    """Launch the Gradio interface."""
    interface = create_interface()
    
    # Launch with sharing disabled by default
    interface.launch(
        server_name="127.0.0.1",
        server_port=7861,
        share=False
    )


if __name__ == "__main__":
    main()
