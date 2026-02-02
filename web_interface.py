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


# Global variables for model and resources
model = None
tokenizer = None
device = None
valid_holds = None
visualizer = None
grade_constraints = None


def initialize_resources():
    """Initialize model, tokenizer, and valid holds."""
    global model, tokenizer, device, valid_holds, visualizer
    
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
    
    print("Resources initialized successfully!")


def generate_and_visualize(
    grade: str,
    valid_holds_only: bool,
    temperature: float
) -> Tuple[Optional[Image.Image], str]:
    """
    Generate a climb path and create visualization.
    
    Hold count is automatically determined based on grade distribution.
    
    Args:
        grade: Climbing grade
        valid_holds_only: Whether to only generate valid holds
        temperature: Sampling temperature
        
    Returns:
        (image, info_text) tuple
    """
    try:
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
            use_reachability=False,  # Disabled - model already trained with reordered paths
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
            f"**Valid Holds Only:** {'Yes' if valid_holds_only else 'No'}",
            f"**Temperature:** {temperature}",
            f"**Hold Range:** {min_holds}-{max_holds} (based on grade distribution)",
            "",
            "**Hold Sequence:**"
        ]
        
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
                
                valid_holds_toggle = gr.Checkbox(
                    value=False,
                    label="Valid Holds Only",
                    info="Only generate holds that exist on MoonBoard 2016 (may cause flat paths)"
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
                temperature_slider
            ],
            outputs=[output_image, info_text]
        )
        
        gr.Markdown(
            """
            ---
            ### About
            
            This tool uses a transformer-based AI model trained on thousands of real MoonBoard climbs.
            The model learns patterns in climbing sequences and generates new, realistic paths.
            
            **Valid Holds Mode:** When enabled, the generator only creates holds at positions where 
            physical boulders exist on the MoonBoard 2016 (141 out of 198 possible positions).
            
            **Tips:**
            - Lower temperature (0.5-0.8) = More conservative, realistic paths
            - Higher temperature (1.2-2.0) = More creative, varied paths
            - Start with default settings and adjust based on results
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
