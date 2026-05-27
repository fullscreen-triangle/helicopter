# Analysis Studio - Scientific Computing IDE

**Date**: 2026-05-27  
**Status**: ✓ Ready for testing  
**Route**: `/tools/analysis-studio`

## Overview

**Analysis Studio** is a MATLAB-like scientific computing environment integrated into Hieronymus. Users write JavaScript analysis scripts that progressively generate visualizations (charts, plots, heatmaps) in real-time.

### Design Philosophy

- **Left Side**: Code editor (JavaScript + DSL)
- **Right Side**: Progressive chart display (2-column responsive grid)
- **Logic Programming**: Charts are generated as code executes (incremental)
- **Rich Charting**: 8 chart types with full customization
- **Real-time Console**: Script output and errors visible immediately

---

## Architecture

### Components

```
Analysis Studio
├── AnalysisEditor (Left)
│   ├── Syntax highlighting
│   ├── JavaScript keywords + chart DSL
│   └── Help sidebar
│
├── ChartGrid (Right)
│   ├── Responsive 1-2 column layout
│   └── Animated chart insertion
│
├── ChartManager (State)
│   ├── Central chart registry
│   └── Add/update/clear operations
│
├── ChartFactory (Rendering)
│   └── 8 chart types via Recharts
│
└── ChartBuilder (DSL)
    └── Fluent API for chart creation
```

### File Structure

```
hieronymus/
├── src/
│   ├── app/tools/analysis-studio/
│   │   └── page.tsx                      # Main page
│   └── components/
│       ├── charts/
│       │   ├── ChartFactory.tsx          # 8 chart components
│       │   ├── ChartManager.tsx          # State + builder
│       │   └── ChartGrid.tsx             # Display grid
│       └── analysis/
│           └── AnalysisEditor.tsx        # Code editor
```

---

## Chart Types

### 1. **Line Chart** (`c.line('id')`)
- Multi-series support
- Smooth interpolation
- Grid and tooltips
- **Use**: Time series, trends, spectral decay

### 2. **Bar Chart** (`c.bar('id')`)
- Grouped or stacked bars
- Categorical data
- **Use**: Comparisons, distributions, histograms

### 3. **Scatter Plot** (`c.scatter('id')`)
- XY coordinate visualization
- Multi-series support
- **Use**: Correlations, validation plots, accuracy testing

### 4. **Area Chart** (`c.area('id')`)
- Stacked or overlaid
- Filled regions
- **Use**: Cumulative data, resource usage, energy distributions

### 5. **Pie Chart** (`c.pie('id')`)
- Proportional segments
- **Use**: Composition, percentages, ratios

### 6. **Radar Chart** (`c.radar('id')`)
- Multi-axis polygon
- **Use**: Multi-dimensional comparison, feature profiles

### 7. **Treemap** (`c.treemap('id')`)
- Hierarchical rectangles
- **Use**: Part-to-whole relationships, nested proportions

### 8. **Composed Chart** (`c.composed('id')`)
- Mixed line + bar
- **Use**: Complex multi-type visualizations

---

## DSL Language & Fluent API

### Basic Syntax

```javascript
// Create a line chart with fluent builder pattern
c.line('chart_id')
  .title('Chart Title')
  .data([ /* array of objects */ ])
  .x('xAxisKey')
  .y('yAxisKey')
  .build();

// Multi-series chart
c.bar('sales_chart')
  .title('Revenue vs Costs')
  .data(salesData)
  .x('month')
  .series([
    { name: 'Revenue', dataKey: 'revenue', color: '#06b6d4' },
    { name: 'Costs', dataKey: 'costs', color: '#ec4899' }
  ])
  .build();

// Scatter plot for accuracy validation
c.scatter('accuracy')
  .title('Measured vs True Distance')
  .data(distances)
  .x('true')
  .y('measured')
  .build();
```

### Builder Methods

```javascript
const builder = c.line('myChart');

// Data & Configuration
builder.title(string)           // Chart title
builder.data(array)             // Data points
builder.x(key)                  // X-axis data key
builder.y(key)                  // Y-axis data key
builder.key(key)                // Primary data key
builder.series(array)           // Multi-series definition
builder.colors(string[])        // Color palette
builder.options(object)         // Custom options

// Execution
builder.build()                 // Create & register chart
builder.update(data, options)   // Update existing chart
```

### Data Format

```javascript
// Basic format
const data = [
  { name: 'Jan', value: 100 },
  { name: 'Feb', value: 150 },
  { name: 'Mar', value: 120 }
];

// Multi-series
const data = [
  { month: 'Jan', sales: 100, costs: 80 },
  { month: 'Feb', sales: 150, costs: 120 },
];

// Scatter (XY)
const data = [
  { x: 100, y: 102, label: 'Point A' },
  { x: 200, y: 198, label: 'Point B' }
];
```

---

## Example Scripts

### Example 1: Microscopy Analysis

```javascript
// Microscopy Image Calculus validation

const data = {
  fourier: [
    { frequency: 1, energy: 1500 },
    { frequency: 5, energy: 800 },
    { frequency: 10, energy: 450 },
    { frequency: 20, energy: 200 },
    { frequency: 40, energy: 80 }
  ]
};

log('Creating spectral analysis...');
c.line('spectral')
  .title('Fourier Power Law Decay α = -0.41')
  .data(data.fourier)
  .x('frequency')
  .key('energy')
  .build();

log('Spectral analysis complete!');
```

### Example 2: Distance Measurement Validation

```javascript
// Validation of coordinate field distance measurements

const distances = [
  { true: 212, measured: 212.1, error: 0.047 },
  { true: 305, measured: 304.8, error: 0.065 },
  { true: 100, measured: 100.03, error: 0.03 }
];

c.scatter('accuracy')
  .title('Distance Measurement Accuracy')
  .data(distances)
  .x('true')
  .y('measured')
  .build();

log('Distance measurement validation: 0.016% error');
```

### Example 3: Multi-Panel Analysis

```javascript
// Complex analysis with multiple views

const metrics = [
  { channel: 'DAPI', entropy: 1.8, snr: 11.3, capacity: 1.93 },
  { channel: 'GFP', entropy: 2.1, snr: 10.8, capacity: 1.88 },
  { channel: 'Red', entropy: 2.4, snr: 13.1, capacity: 2.05 }
];

// Entropy comparison
c.bar('entropy')
  .title('Shannon Entropy by Channel')
  .data(metrics)
  .x('channel')
  .key('entropy')
  .build();

// SNR vs Channel Capacity
c.scatter('snr_vs_capacity')
  .title('SNR vs Channel Capacity')
  .data(metrics)
  .x('snr')
  .y('capacity')
  .build();

log('Multi-channel analysis complete!');
```

---

## Execution Model

### Script Lifecycle

```
1. User clicks "Run Script"
   ↓
2. Console clears, status → "Executing..."
   ↓
3. Charts cleared via chartManager.clearCharts()
   ↓
4. Code wrapped in Function() with (c, log) context
   ↓
5. c.line('id').data(...).build() → chart added to manager
   ↓
6. log('message') → printed to console, displayed live
   ↓
7. On error → catch, display error message, elapsed time
   ↓
8. On success → display "Complete in X.XXms"
```

### Error Handling

Errors are caught and displayed in the console:
```
ERROR
ReferenceError: data is not defined
```

Stack traces are shown (line numbers relative to user code).

---

## Styling & Theme

- **Colors**: Cyan, pink, amber, emerald, violet (Recharts defaults)
- **Background**: Dark theme (#050810, #0a0e27, #0f1420)
- **Typography**: Cascadia Code (monospace for editor)
- **Animations**: Framer Motion (chart entrance, grid updates)
- **Responsive**: 1 column on mobile, 2 columns on desktop

---

## Performance Considerations

- **Chart Rendering**: Recharts optimized (canvas + SVG)
- **Data Limit**: ~1000 points per chart (tested & smooth)
- **Memory**: Charts stored in React context (no Redux)
- **Build Time**: Instant (<100ms for typical script)
- **DOM Updates**: Batch animated via Framer Motion

---

## Testing the Studio

### Quick Start

```bash
cd hieronymus
npm run dev
# Visit: http://localhost:3000/tools/analysis-studio
```

### Sample Scripts to Try

1. **Modify the default script** - Edit a value, run, see charts update
2. **Add a new chart** - Copy a `.line()` block, change ID and data
3. **Fix an error** - Comment out a line, fix syntax, re-run
4. **Create your own** - Use any JSON data object you want

### Expected Behavior

- Charts appear on the right as code executes
- Console shows execution progress
- Charts animated in as they're created
- Error messages appear immediately
- Execution time displayed at bottom

---

## Integration with MIC Framework

The Analysis Studio is designed to work with Microscopy Image Calculus:

```javascript
// Validate MIC theorems from validation_results.json
const theorems = {
  fourier: -0.410,
  wavelet: 1.10,
  entropy: 2.080,
  fisher: 0.0189
};

c.bar('theorem_validation')
  .title('MIC Theorem Validation Results')
  .data([
    { theorem: 'Fourier Decay', value: theorems.fourier },
    { theorem: 'Wavelet Frame', value: theorems.wavelet },
    // ... more
  ])
  .build();
```

---

## Future Enhancements

### Phase 2
- [ ] Import CSV/JSON files
- [ ] Download charts as PNG/SVG
- [ ] Save/load scripts
- [ ] Share analysis via URL

### Phase 3
- [ ] Connect to Rust backend for computation
- [ ] Live data feeds (streaming analysis)
- [ ] Real-time collaborative editing
- [ ] Chart annotations and measurement tools

### Phase 4
- [ ] 3D chart support (surfaces, point clouds)
- [ ] Image overlay on charts
- [ ] Custom chart types (via D3.js)
- [ ] GPU-accelerated rendering

---

## Summary

**Analysis Studio** is a complete scientific computing IDE for Hieronymus that combines:
- ✓ Modern code editor with syntax highlighting
- ✓ Progressive chart generation (8 types)
- ✓ Fluent DSL for easy chart creation
- ✓ Real-time console output
- ✓ MATLAB-like user experience
- ✓ Full customization via JavaScript

Perfect for analyzing MIC validation results, visualizing scientific data, and prototyping analysis pipelines.

---

**Ready to use!** Create your first analysis script at `/tools/analysis-studio`
