# Climb Path Visualization Guide

This guide explains how to visualize generated climbing paths on the MoonBoard 2016 background image.

## Overview

The project includes two visualization methods:

1. **Python Script** - Generates static images with paths overlaid on the moonboard background
2. **React Frontend** - Interactive web interface with real-time path visualization

## Method 1: Python Visualization Script

### Features
- Overlays climbing paths on the actual MoonBoard 2016 background image
- Circles holds with color-coded markers:
  - 🟢 **Green** - Start hold
  - 🔵 **Blue** - Middle holds
  - 🔴 **Red** - End hold
- Numbered holds showing the sequence
- Dashed lines connecting holds
- Grade and hold count labels

### Usage

#### Step 1: Generate Paths
First, generate some climbing paths using the model:

```bash
python backend/training/generate_paths.py --grade 7A --num_samples 5 --save_json generated_paths.json --visualize
```

#### Step 2: Visualize on MoonBoard Image
Run the visualization script:

```bash
python visualize_path.py --input generated_paths.json --output visualizations
```

Or use the batch file:

```bash
visualize_path.bat
```

#### Options
- `--background` - Path to moonboard background image (default: `data/moonboard2016Background.jpg`)
- `--input` - Path to JSON file with generated paths (required)
- `--output` - Output directory for visualizations (default: `visualizations`)
- `--circle_radius` - Radius of circles around holds (default: 25)
- `--line_thickness` - Thickness of connecting lines (default: 4)
- `--no_numbers` - Hide hold numbers

#### Example with Custom Options
```bash
python visualize_path.py --input my_paths.json --output my_viz --circle_radius 30 --line_thickness 6
```

### Output
The script generates one PNG image per path in the output directory, named:
- `path_1_7A.png`
- `path_2_7A.png`
- etc.

## Method 2: React Frontend Visualization

### Features
- Interactive web interface
- Real-time path generation and visualization
- Uses actual MoonBoard 2016 background image
- Responsive canvas rendering
- Grade selection and advanced controls
- Temperature and hold count customization

### Setup

#### Step 1: Install Frontend Dependencies
```bash
cd frontend
npm install
```

#### Step 2: Copy MoonBoard Image
The moonboard background image should be in `frontend/public/data/moonboard2016Background.jpg`.
This is automatically copied during setup.

#### Step 3: Start Backend Server
In one terminal:

```bash
start_server.bat
```

Or manually:
```bash
cd backend/api
python server.py
```

The API server will start at `http://localhost:8000`

#### Step 4: Start Frontend
In another terminal:

```bash
start_frontend.bat
```

Or manually:
```bash
cd frontend
npm run dev
```

The frontend will start at `http://localhost:3000`

### Using the Interface

1. **Select Grade** - Choose difficulty level (6B to 8B+)
2. **Advanced Options** (optional):
   - **Temperature** - Controls randomness (0.1 = deterministic, 2.0 = very random)
   - **Hold Count Range** - Min/max number of holds in the path
3. **Generate Path** - Click to generate a new climbing route
4. **View Path** - The path appears overlaid on the moonboard with:
   - Green circle = start hold
   - Blue circles = middle holds
   - Red circle = end hold
   - Yellow dashed lines connecting holds
   - Numbers showing the sequence

## MoonBoard Grid Coordinates

The MoonBoard 2016 uses the following coordinate system:

- **Columns**: A-K (0-10 in code)
- **Rows**: 1-18 (bottom to top)

Example coordinates:
- `[0, 1]` = Bottom-left (A1)
- `[10, 18]` = Top-right (K18)
- `[5, 10]` = Middle (F10)

## Coordinate Mapping

The visualization scripts automatically map grid coordinates to pixel positions:

```python
# Grid coordinate (x, y) where:
# x = 0-10 (columns A-K)
# y = 1-18 (rows, bottom to top)

# Pixel position calculation:
pixel_x = (x + 0.5) * cell_width
pixel_y = (18 - y + 0.5) * cell_height  # Y is inverted
```

## Troubleshooting

### Python Script Issues

**Error: "Could not load image"**
- Ensure `data/moonboard2016Background.jpg` exists
- Check the `--background` path is correct

**Error: "No paths found in input file"**
- Verify the JSON file contains a `paths` array
- Check the file was generated correctly

### Frontend Issues

**Moonboard background not showing**
- Check browser console for image loading errors
- Verify `frontend/public/data/moonboard2016Background.jpg` exists
- Try hard refresh (Ctrl+F5)

**API connection errors**
- Ensure backend server is running on port 8000
- Check CORS settings in `backend/api/server.py`
- Verify frontend proxy configuration in `vite.config.js`

**Path not generating**
- Check that model checkpoint exists at `checkpoints/climb_path_cpu/best.pt`
- Review backend server logs for errors
- Ensure the selected grade is valid

## API Endpoints

The backend provides these endpoints:

- `GET /` - API info
- `GET /api/health` - Health check
- `GET /api/grades` - List available grades
- `POST /api/generate` - Generate a climbing path

### Generate Path Request
```json
{
  "grade": "7A",
  "temperature": 1.0,
  "min_holds": 3,
  "max_holds": 30,
  "use_constraints": true
}
```

### Generate Path Response
```json
{
  "path": [[5, 4], [6, 6], [7, 8], [6, 10], [7, 12], [6, 15], [6, 17]],
  "grade": "7A",
  "num_holds": 7,
  "quality_score": null
}
```

## Examples

### Generate and Visualize a 7A+ Route
```bash
# Generate paths
python backend/training/generate_paths.py --grade 7A+ --num_samples 3 --temperature 1.2 --save_json my_7a_plus.json

# Visualize
python visualize_path.py --input my_7a_plus.json --output viz_7a_plus
```

### Generate Easy Routes (6B)
```bash
python backend/training/generate_paths.py --grade 6B --num_samples 10 --temperature 0.8 --max_holds 20 --save_json easy_routes.json
python visualize_path.py --input easy_routes.json --output easy_viz
```

### Generate Hard Routes (8A)
```bash
python backend/training/generate_paths.py --grade 8A --num_samples 5 --temperature 1.5 --min_holds 15 --save_json hard_routes.json
python visualize_path.py --input hard_routes.json --output hard_viz
```

## Next Steps

- Add path quality scoring
- Implement path validation
- Add difficulty prediction
- Export paths for training
- Add path comparison tools
- Create batch visualization scripts
