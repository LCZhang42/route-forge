# Frontend

Web interface for the Climb Path Generator.

## Features (Planned)

- 🎯 **Interactive Grade Selection** - Choose difficulty level (6B to 8B+)
- 🧗 **MoonBoard Visualization** - Visual representation of the climbing wall
- 🎨 **Custom Constraints** - Set start/end positions
- 📊 **Path Preview** - See generated routes with hold sequences
- 💾 **Save/Export** - Export routes for training

## Tech Stack

- **React** - UI framework
- **TailwindCSS** - Styling
- **Lucide React** - Icons
- **Canvas/Three.js** - MoonBoard visualization
- **Axios** - API communication

## Setup

```bash
cd frontend
npm install
npm run dev
```

## Structure (To Be Created)

```
frontend/
├── src/
│   ├── components/
│   │   ├── MoonBoard.jsx      # Interactive board visualization
│   │   ├── GradeSelector.jsx  # Difficulty selector
│   │   ├── PathDisplay.jsx    # Show generated path
│   │   └── Controls.jsx       # Generation controls
│   ├── api/
│   │   └── client.js          # API client
│   ├── App.jsx
│   └── main.jsx
├── package.json
└── vite.config.js
```

## API Integration

The frontend will communicate with the backend API:

```javascript
// Generate a new path
POST /api/generate
{
  "grade": "7A",
  "constraints": {
    "start_position": [5, 4],
    "end_position": [6, 17]
  }
}

// Response
{
  "path": [[5,4], [6,6], [7,8], ..., [6,17]],
  "grade": "7A",
  "quality_score": 0.85
}
```
