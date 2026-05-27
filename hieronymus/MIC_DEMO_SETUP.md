# MIC Demo Tool - Setup & Architecture

**Date**: 2026-05-27  
**Status**: ✓ Ready for testing

## Overview

Added a new interactive sandbox tool to Hieronymus for the Microscopy Image Calculus (MIC) framework. The tool allows users to write analysis scripts, compile them to GPU shaders (in TypeScript), and visualize 3D results in real-time.

## File Structure

```
hieronymus/
├── src/
│   ├── app/
│   │   └── tools/
│   │       └── mic-demo/
│   │           └── page.tsx           # Main page component
│   └── components/
│       └── mic-demo/
│           ├── SceneViewer.tsx        # 3D visualization (React Three Fiber)
│           ├── CodeEditor.tsx         # Code editor with syntax highlighting
│           └── ControlPanel.tsx       # Visualization mode selector + results
└── public/
    └── (cell structures can be added here)
```

## Components

### 1. **Main Page** (`/tools/mic-demo/page.tsx`)
- **Layout**: Horizontal split screen (50% code, 50% visualization)
- **Features**:
  - TypeScript DSL code editor
  - Run button to compile/execute
  - Real-time status updates
  - Performance metrics display
- **State Management**: 
  - Code content
  - Analysis results (distance, scale field, etc.)
  - Visualization mode selection

### 2. **SceneViewer** (`SceneViewer.tsx`)
- **Technology**: React Three Fiber + Three.js
- **Features**:
  - 3D point cloud visualization
  - Grid and axes for reference
  - OrbitControls for camera manipulation
  - Real-time color-mapped visualization
- **Visualization Modes**:
  - `scale-field`: Heatmap of metric scales (blue → red)
  - `segmentation`: Binary structure highlighting (green)
  - `distance`: Gradient-based distance visualization

### 3. **CodeEditor** (`CodeEditor.tsx`)
- **Features**:
  - Line numbers
  - Syntax highlighting (keywords, braces)
  - Tab indentation support
  - Keyword reference panel on right
- **Language**: Simplified DSL for MIC operations
  ```typescript
  analyze {
    load channel: "synthetic"
    estimate scale_field
    visualize as: heatmap
    measure_distance from: [64, 64] to: [192, 192]
  }
  ```

### 4. **ControlPanel** (`ControlPanel.tsx`)
- **Features**:
  - Visualization mode selector (3 buttons)
  - Results display (distance, scale field statistics)
  - Performance metrics (compilation time)
  - Help text when idle

## TypeScript DSL Language

The editor accepts a simple domain-specific language for MIC operations:

```typescript
analyze {
  load channel: <channel_name>
  estimate <operation>
  visualize as: <mode>
  measure_distance from: [x1, y1] to: [x2, y2]
}
```

**Available Channels**:
- `synthetic` - Generated test image
- `dapi` - Nucleus stain
- `gfp` - Protein fluorescence
- `red` - Red channel

**Available Operations**:
- `scale_field` - Estimate metric scales from spectral analysis
- `segmentation` - Compute structure boundary
- `distance` - Measure metric-grounded distance

**Visualization Modes**:
- `heatmap` - Color-mapped values
- `segmentation` - Binary mask overlay
- `distance` - Gradient visualization

## Current Implementation

### Phase 1 (Complete ✓)
- [x] Split-screen layout with code editor
- [x] 3D visualization viewport (Three.js)
- [x] TypeScript code editor with syntax highlighting
- [x] Control panel with mode selector
- [x] Dummy data generation for testing visualization
- [x] Result display (distance, statistics)

### Phase 2 (Next: TypeScript Compiler)
- [ ] Implement TypeScript DSL parser
- [ ] Compile DSL to GLSL shaders
- [ ] WebGL/WGPU shader execution
- [ ] Real-time visualization updates
- [ ] Sample cell structure loading from database

### Phase 3 (Integration)
- [ ] Connection to Rust backend (HTTP/WebSocket)
- [ ] Validation result comparison
- [ ] More complex analysis operations
- [ ] Project/workspace management

## Running the Demo

1. **Start the development server**:
   ```bash
   cd hieronymus
   npm run dev
   ```

2. **Navigate to the tool**:
   - Click "MIC Demo" in the navbar
   - Or go directly to: `http://localhost:3000/tools/mic-demo`

3. **Test the interface**:
   - Edit the code on the left
   - Click "Run Analysis" button
   - View results in 3D visualization on the right
   - Switch between visualization modes in control panel

## Architecture Notes

### State Flow
```
User edits code
    ↓
Click "Run Analysis"
    ↓
TypeScript compiler (to be implemented)
    ↓
Generate GLSL shaders
    ↓
Execute on GPU (WebGL/WGPU)
    ↓
Return results (distance, statistics)
    ↓
Update visualization in 3D viewport
    ↓
Display metrics in control panel
```

### Visualization Pipeline
```
Float32Array (scale field data)
    ↓
Map to RGB colors (0.0 → 1.0 → heatmap)
    ↓
Create BufferAttribute for Three.js
    ↓
Render as point cloud with PointsMaterial
    ↓
Interactive camera with OrbitControls
```

## Dependencies (Already Installed)

- `next@14.2.0` - React framework
- `react@18.3.0` - UI library
- `@react-three/fiber@8.15.19` - React renderer for Three.js
- `@react-three/drei@9.88.17` - Helpful utilities for Three.js
- `three@0.160.0` - 3D graphics library
- `framer-motion@10.0.1` - Animation library
- `tailwindcss@3.2.7` - Styling

## Next Steps

1. **Test the UI** (no compilation yet)
   - Verify all components render
   - Test layout responsiveness
   - Confirm interaction works

2. **Implement TypeScript Compiler**
   - Parse DSL code
   - Generate simple GLSL shaders
   - Execute on GPU
   - Return dummy data

3. **Add Real Algorithms**
   - Call Rust backend for computations
   - Stream results to visualization
   - Implement proper scale field estimation

4. **Database Integration**
   - Load sample 3D cell structures
   - Support TIFF stack import
   - Cache visualization data

## Files Modified

- **NavbarApp.tsx**: Added `/tools/mic-demo` link to navigation

## Files Created

- `src/app/tools/mic-demo/page.tsx`
- `src/components/mic-demo/SceneViewer.tsx`
- `src/components/mic-demo/CodeEditor.tsx`
- `src/components/mic-demo/ControlPanel.tsx`

---

**Ready to test!** The web demo UI is complete and ready for compiler implementation.
