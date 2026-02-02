"""
Visualize generated climb paths on the MoonBoard 2016 background image.

This script overlays climbing paths onto the moonboard image by drawing
circles around the holds and connecting them with lines.
"""

import cv2
import numpy as np
import json
import argparse
from pathlib import Path
from typing import List, Tuple, Optional


class MoonBoardVisualizer:
    """Visualize climb paths on MoonBoard 2016 background."""
    
    # MoonBoard 2016 grid specifications
    COLUMNS = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K']
    ROWS = list(range(1, 19))  # 1-18
    
    def __init__(self, background_path: str):
        """
        Initialize visualizer with background image.
        
        Args:
            background_path: Path to moonboard background image
        """
        self.background = cv2.imread(background_path)
        if self.background is None:
            raise ValueError(f"Could not load image from {background_path}")
        
        self.height, self.width = self.background.shape[:2]
        
        # Based on reference implementation (andrew-houghton/moon-board-climbing)
        # The moonboard image is 650px x 1000px in the reference
        # Grid positioning: padding-top: 61px, margin-left: 68px, margin-right: 30px
        # Grid cells: 50px x 50px, 11 columns x 18 rows
        
        # Scale these values to match our actual image dimensions
        scale_x = self.width / 650.0
        scale_y = self.height / 1000.0
        
        # Grid offsets from reference implementation
        self.grid_left = int(68 * scale_x)  # margin-left: 68px
        self.grid_top = int(61 * scale_y)   # padding-top: 61px
        
        # Cell dimensions from reference (50px x 50px)
        self.cell_width = 50 * scale_x
        self.cell_height = 50 * scale_y
        
    def coordinate_to_pixel(self, x: int, y: int) -> Tuple[int, int]:
        """
        Convert grid coordinates to pixel coordinates.
        
        Args:
            x: Column index (0-10 for A-K)
            y: Row index (1-18)
            
        Returns:
            (pixel_x, pixel_y) tuple
        """
        # Grid-based positioning matching reference implementation
        # Each cell is positioned at grid_left + (column * cell_width)
        # Y coordinate: row 18 is at top (index 0), row 1 is at bottom (index 17)
        pixel_x = int(self.grid_left + x * self.cell_width + self.cell_width / 2)
        pixel_y = int(self.grid_top + (18 - y) * self.cell_height + self.cell_height / 2)
        
        return pixel_x, pixel_y
    
    def draw_path(
        self,
        holds: List[Tuple[int, int]],
        grade: str,
        output_path: str,
        circle_radius: int = 25,
        line_thickness: int = 4,
        show_numbers: bool = True,
    ):
        """
        Draw a climb path on the moonboard image.
        
        Args:
            holds: List of (x, y) hold coordinates
            grade: Grade string for display
            output_path: Path to save output image
            circle_radius: Radius of circles around holds
            line_thickness: Thickness of connecting lines
            show_numbers: Whether to show hold numbers
        """
        # Create a copy of the background
        img = self.background.copy()
        
        # Define colors (BGR format for OpenCV)
        color_start = (0, 255, 0)      # Green for start
        color_mid = (255, 100, 0)      # Blue for middle holds
        color_end = (0, 0, 255)        # Red for end
        color_line = (255, 200, 0)     # Cyan for connecting lines
        color_text = (255, 255, 255)   # White for text
        
        # Draw connecting lines first (so they appear behind circles)
        if len(holds) > 1:
            for i in range(len(holds) - 1):
                x1, y1 = holds[i]
                x2, y2 = holds[i + 1]
                
                pixel1 = self.coordinate_to_pixel(x1, y1)
                pixel2 = self.coordinate_to_pixel(x2, y2)
                
                # Draw dashed line
                self._draw_dashed_line(img, pixel1, pixel2, color_line, line_thickness)
        
        # Draw circles around holds
        for idx, (x, y) in enumerate(holds):
            pixel_x, pixel_y = self.coordinate_to_pixel(x, y)
            
            # Choose color based on position
            if idx == 0:
                color = color_start
            elif idx == len(holds) - 1:
                color = color_end
            else:
                color = color_mid
            
            # Draw outer circle (thicker)
            cv2.circle(img, (pixel_x, pixel_y), circle_radius, color, line_thickness)
            
            # Draw inner filled circle (semi-transparent)
            overlay = img.copy()
            cv2.circle(overlay, (pixel_x, pixel_y), circle_radius - 5, color, -1)
            cv2.addWeighted(overlay, 0.3, img, 0.7, 0, img)
            
            # Draw hold number
            if show_numbers:
                text = str(idx + 1)
                font = cv2.FONT_HERSHEY_SIMPLEX
                font_scale = 1.0
                font_thickness = 2
                
                # Get text size for centering
                (text_width, text_height), baseline = cv2.getTextSize(
                    text, font, font_scale, font_thickness
                )
                
                text_x = pixel_x - text_width // 2
                text_y = pixel_y + text_height // 2
                
                # Draw text with black outline for visibility
                cv2.putText(img, text, (text_x, text_y), font, font_scale, 
                           (0, 0, 0), font_thickness + 2, cv2.LINE_AA)
                cv2.putText(img, text, (text_x, text_y), font, font_scale, 
                           color_text, font_thickness, cv2.LINE_AA)
        
        # Add grade label
        self._add_grade_label(img, grade, len(holds))
        
        # Save output
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(output_path), img)
        
        print(f"Saved visualization to {output_path}")
        
        return img
    
    def _draw_dashed_line(
        self,
        img: np.ndarray,
        pt1: Tuple[int, int],
        pt2: Tuple[int, int],
        color: Tuple[int, int, int],
        thickness: int,
        dash_length: int = 15,
    ):
        """Draw a dashed line between two points."""
        x1, y1 = pt1
        x2, y2 = pt2
        
        # Calculate line length and angle
        dx = x2 - x1
        dy = y2 - y1
        length = np.sqrt(dx**2 + dy**2)
        
        if length == 0:
            return
        
        # Normalize direction
        dx /= length
        dy /= length
        
        # Draw dashes
        current_length = 0
        draw = True
        
        while current_length < length:
            if draw:
                start_x = int(x1 + dx * current_length)
                start_y = int(y1 + dy * current_length)
                end_length = min(current_length + dash_length, length)
                end_x = int(x1 + dx * end_length)
                end_y = int(y1 + dy * end_length)
                cv2.line(img, (start_x, start_y), (end_x, end_y), color, thickness, cv2.LINE_AA)
            
            current_length += dash_length
            draw = not draw
    
    def _add_grade_label(self, img: np.ndarray, grade: str, num_holds: int):
        """Add grade and hold count label to image."""
        text = f"Grade: {grade} | Holds: {num_holds}"
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.75  # Reduced by half from 1.5
        font_thickness = 2  # Reduced from 3
        
        # Get text size
        (text_width, text_height), baseline = cv2.getTextSize(
            text, font, font_scale, font_thickness
        )
        
        # Position at bottom center
        x = (self.width - text_width) // 2
        y = self.height - 25
        
        # Draw background rectangle
        padding = 15
        cv2.rectangle(
            img,
            (x - padding, y - text_height - padding),
            (x + text_width + padding, y + baseline + padding),
            (0, 0, 0),
            -1
        )
        
        # Draw text
        cv2.putText(img, text, (x, y), font, font_scale, (255, 255, 255), 
                   font_thickness, cv2.LINE_AA)
    
    def draw_multiple_paths(
        self,
        paths_data: List[dict],
        output_dir: str,
        **kwargs
    ):
        """
        Draw multiple paths and save as separate images.
        
        Args:
            paths_data: List of dicts with 'holds' and 'grade' keys
            output_dir: Directory to save output images
            **kwargs: Additional arguments for draw_path
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        for idx, path_data in enumerate(paths_data):
            holds = path_data['holds']
            grade = path_data['grade']
            
            output_path = output_dir / f"path_{idx+1}_{grade}.png"
            self.draw_path(holds, grade, str(output_path), **kwargs)


def main():
    parser = argparse.ArgumentParser(
        description='Visualize climb paths on MoonBoard background'
    )
    
    parser.add_argument(
        '--background',
        type=str,
        default='data/moonboard2016Background.jpg',
        help='Path to moonboard background image'
    )
    parser.add_argument(
        '--input',
        type=str,
        required=True,
        help='Path to JSON file with generated paths'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='visualizations',
        help='Output directory for visualizations'
    )
    parser.add_argument(
        '--circle_radius',
        type=int,
        default=25,
        help='Radius of circles around holds'
    )
    parser.add_argument(
        '--line_thickness',
        type=int,
        default=4,
        help='Thickness of connecting lines'
    )
    parser.add_argument(
        '--no_numbers',
        action='store_true',
        help='Hide hold numbers'
    )
    
    args = parser.parse_args()
    
    # Load paths from JSON
    with open(args.input, 'r') as f:
        data = json.load(f)
    
    paths = data.get('paths', [])
    
    if not paths:
        print("No paths found in input file")
        return
    
    print(f"Loaded {len(paths)} paths from {args.input}")
    
    # Initialize visualizer
    visualizer = MoonBoardVisualizer(args.background)
    
    # Draw all paths
    visualizer.draw_multiple_paths(
        paths,
        args.output,
        circle_radius=args.circle_radius,
        line_thickness=args.line_thickness,
        show_numbers=not args.no_numbers,
    )
    
    print(f"\nVisualization complete! Images saved to {args.output}/")


if __name__ == '__main__':
    main()
