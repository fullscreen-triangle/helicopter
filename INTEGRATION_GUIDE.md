# SCOPE + Analysis Studio Integration Guide

**Status**: Integration complete, ready for testing  
**Date**: 2026-05-27

## Overview

This guide walks through setting up and testing the integrated SCOPE metalanguage with the Analysis Studio web tool.

### Architecture

```
┌─────────────────────────────────────────────┐
│     Analysis Studio (Next.js/React)         │
│  - JavaScript DSL mode (existing)           │
│  - SCOPE mode (new)                         │
│  - Program selector                         │
│  - Phase selector (for nuclear_separation)  │
└────────────┬────────────────────────────────┘
             │
             │ HTTP/REST API
             │
┌────────────▼────────────────────────────────┐
│    SCOPE Backend (Flask/Python)             │
│  - Program registry                         │
│  - Five-phase execution                     │
│  - Result serialization                     │
└────────────┬────────────────────────────────┘
             │
      ┌──────┴──────┐
      │             │
  ┌───▼──┐    ┌────▼────┐
  │Python│    │ SciPy/  │
  │SCOPE │    │NumPy    │
  │DSL   │    │ (spectral)
  └──────┘    └─────────┘
```

## Step 1: Install Dependencies

### For SCOPE Backend

```bash
cd turbine/scope/server
pip install -r requirements.txt
```

### For Analysis Studio (Already installed)

The Hieronymus project uses Next.js, which is already configured.

## Step 2: Start SCOPE Backend Server

```bash
cd turbine/scope/server

# Option 1: Direct Python execution
python -m flask --app app run

# Option 2: Using Python module
python -c "from turbine.scope.server import create_app; app = create_app(debug=True); app.run(host='0.0.0.0', port=5000)"
```

Expected output:
```
 * Running on http://127.0.0.1:5000
 * Debug mode: on
```

## Step 3: Start Analysis Studio Web Tool

In a separate terminal:

```bash
cd hieronymus
npm run dev
```

Expected output:
```
> next dev
▲ Next.js 14.x.x
- Local:        http://localhost:3000
```

## Step 4: Access the Application

1. Open browser: **http://localhost:3000/tools/analysis-studio**
2. You should see the Analysis Studio with two new features:
   - **Mode selector**: "JavaScript" (default) and "SCOPE" buttons
   - **SCOPE controls**: When SCOPE mode is selected

## Step 5: Test JavaScript Mode (Existing Functionality)

1. Mode should already be set to "JavaScript"
2. Click **Run Script** button
3. You should see:
   - 4 charts appear on the right (spectral, scale distribution, distance accuracy, error)
   - Console output on the left showing execution progress
   - Green "Complete in X.XXms" message

## Step 6: Test SCOPE Mode

### Prerequisites
- SCOPE backend must be running (Step 2)
- Browser console should show no errors when switching to SCOPE mode

### Test Steps

1. **Switch to SCOPE mode**:
   - Click the "SCOPE" button in the header
   - If backend is running, it should be enabled (not grayed out)
   - If disabled, check that Flask backend is running on port 5000

2. **Select a program**:
   - A dropdown should appear showing available programs
   - Should see "nuclear_separation_dynamics" (the example program)

3. **Select cell cycle phase** (appears for nuclear_separation):
   - Three buttons should appear: PROPHASE, METAPHASE, ANAPHASE
   - Select one (PROPHASE is default)

4. **Review program details**:
   - Depth: 1000
   - Field: 100.0×100.0 µm
   - Resolution: 0.1 µm/px
   - Morphisms: nucleus_pair_measurement, membrane_boundary

5. **Execute**:
   - Click **Run** button
   - Console output should show:
     - "Executing SCOPE program: nuclear_separation_dynamics"
     - "Cell cycle phase: PROPHASE"
     - "Generated 1000 timing events"
     - "Generated synthetic frame: 1024×1024"
     - "Execution complete in XXms"
     - Distance measurement result
     - Uncertainty bounds
     - Position in world-space
     - S-entropy values

## Expected Console Output

```
Executing SCOPE program: nuclear_separation_dynamics
Cell cycle phase: PROPHASE
Generated 1000 timing events
Generated synthetic frame: 1024×1024
✓ Execution complete in 234.5ms
Structure: separation_vector
Distance: 8.5e-06m
Uncertainty: ±1.4e-07m
Position: (0.039, 0.164, -2.000)
S-entropy: S_k=0.500, S_t=1e-06, S_e=0.500
```

## Troubleshooting

### SCOPE Mode Is Greyed Out

**Cause**: Backend not running or not accessible

**Solution**:
```bash
# Check if Flask is running
lsof -i :5000

# If not running, start it
cd turbine/scope/server
python -m flask --app app run --port 5000
```

### Browser Console Shows CORS Errors

**Cause**: Flask-CORS not installed properly

**Solution**:
```bash
pip install flask-cors
```

### Execution Fails with "No timing events provided"

**Cause**: Client-side timing event generation failed

**Solution**: Check browser console for JavaScript errors. The `generateTimingEvents` function in `scope-client.ts` should work without additional dependencies.

### Results Show NaN or Invalid Values

**Cause**: NumPy/SciPy floating-point issue

**Solution**:
```bash
# Ensure correct versions
pip install numpy==1.24.3 scipy==1.11.2
```

## Testing Workflow

### Full End-to-End Test

```bash
# Terminal 1: Start backend
cd turbine/scope/server
python -m flask --app app run

# Terminal 2: Start frontend
cd hieronymus
npm run dev

# Terminal 3: Manual testing (curl)
curl http://localhost:5000/health
curl http://localhost:5000/programs
```

### Unit Tests (Python)

```bash
# Test SCOPE phases individually
cd turbine
python -m pytest scope/tests/ -v
```

(Note: tests not yet implemented; this is for future use)

### Load Testing

```bash
# Test batch execution with 10 streams
python -c "
from turbine.scope.programs.nuclear_separation import run_example
run_example()
"
```

## Integration Points

### Analysis Studio → SCOPE Backend

**File**: `hieronymus/src/app/tools/analysis-studio/page.tsx`

**Flow**:
1. User selects SCOPE mode
2. Frontend calls `SCOPEClient.healthCheck()` → `/health`
3. If available, calls `listPrograms()` → `/programs`
4. User selects program and phase
5. Click Run → `execute()` → `/execute` endpoint
6. Results displayed in console and charts (if applicable)

### Client Library

**File**: `hieronymus/src/lib/scope-client.ts`

**Classes**:
- `SCOPEClient`: HTTP wrapper for all endpoints
- Helper functions: `generateTimingEvents()`, `generateSyntheticFrame()`, `SCOPEClient.encodeFrame()`, `SCOPEClient.decodeResult()`

### Backend Server

**File**: `turbine/scope/server/app.py`

**Endpoints**:
- `GET /health` — Health check
- `GET /programs` — List available programs
- `GET /programs/<id>` — Get program details
- `POST /execute` — Execute single program
- `POST /execute-batch` — Execute multiple streams
- `POST /encode-frame` — Helper for frame encoding

## Data Flow

### Execution Request

```json
{
  "program_id": "nuclear_separation_dynamics",
  "timing_events": [
    {"delta_p": -1.2e-6, "channel_id": 0, "intensity": 100},
    ...
  ],
  "frame": {
    "data": "base64-encoded-array",
    "shape": [1024, 1024],
    "dtype": "float32"
  }
}
```

### Execution Response

```json
{
  "success": true,
  "result": {
    "structure": "separation_vector",
    "position": {"x": 0.039, "y": 0.164, "z": -2.0},
    "distance": 8.5e-6,
    "uncertainty": 1.4e-7,
    "s_entropy": {"S_k": 0.5, "S_t": 1e-6, "S_e": 0.5},
    "partition_state": {"n": 1000, "ℓ": 10, "m": 42, "s": 1}
  },
  "timing_ms": 234.5
}
```

## Next Steps

### Immediate (Testing)
- [ ] Verify both JavaScript and SCOPE modes work
- [ ] Check console output formatting
- [ ] Validate timing values are reasonable

### Short-term (Enhancement)
- [ ] Add chart generation for SCOPE results (distance vs. uncertainty plots)
- [ ] Implement batch execution UI
- [ ] Add program parameters configuration
- [ ] Cache program list to reduce API calls

### Medium-term (Features)
- [ ] Connect to real image databases (BBBC, Allen Cell)
- [ ] Multi-frame time-lapse support
- [ ] Save/load analysis scripts
- [ ] Export results as JSON/CSV

### Long-term (Production)
- [ ] Rust backend (GPU-accelerated spectral pipeline)
- [ ] WebSocket for streaming results
- [ ] Authentication and user workspaces
- [ ] Database integration for result storage

## Reference

**Documentation**:
- `SCOPE_IMPLEMENTATION_STATUS.md` — Full SCOPE implementation details
- `turbine/scope/README.md` — Python DSL user guide
- `hieronymus/publications/sources/scope-metalanguage.tex` — Formal specification

**Key Files**:
- Backend: `turbine/scope/server/app.py`
- Client: `hieronymus/src/lib/scope-client.ts`
- UI: `hieronymus/src/app/tools/analysis-studio/page.tsx`
- Example program: `turbine/scope/programs/nuclear_separation.py`

---

**Ready to test!** Start with Step 1 and work through the checklist.
