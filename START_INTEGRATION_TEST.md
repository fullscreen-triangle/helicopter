# Quick Start: SCOPE + Analysis Studio Integration Test

**⏱️ Time required**: ~5 minutes to set up, 2 minutes to test

## One-Command Quick Start

### Option 1: On Windows (PowerShell)

```powershell
# Terminal 1: Start SCOPE Backend
cd turbine/scope/server
pip install -r requirements.txt
python -m flask --app app run --port 5000

# Terminal 2: Start Analysis Studio
cd hieronymus
npm run dev
```

Then open: **http://localhost:3000/tools/analysis-studio**

### Option 2: On macOS/Linux (Bash)

```bash
# Terminal 1: Backend
cd turbine/scope/server
pip install -r requirements.txt
python -m flask --app app run --port 5000 &

# Terminal 2: Frontend
cd hieronymus
npm run dev
```

Then open: **http://localhost:3000/tools/analysis-studio**

## What You'll See

### Phase 1: JavaScript Mode (Should Already Work)
- Left: Code editor with JavaScript DSL
- Right: Progressive chart grid
- Click "Run Script" → 4 charts appear
- Console shows: "Creating spectral analysis...", "Analysis complete!"

### Phase 2: SCOPE Mode (New)
1. Click "SCOPE" button in header
2. Select program: "nuclear_separation_dynamics"
3. Select phase: "PROPHASE" (default)
4. Click "Run"
5. Console shows:
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

## Verification Checklist

- [ ] JavaScript mode works (4 charts appear)
- [ ] SCOPE button appears in header
- [ ] SCOPE button is enabled (not greyed out)
- [ ] Program dropdown shows "nuclear_separation_dynamics"
- [ ] Phase buttons appear (PROPHASE, METAPHASE, ANAPHASE)
- [ ] Clicking "Run" executes without errors
- [ ] Console shows SCOPE execution output
- [ ] Distance measurement appears in output
- [ ] S-entropy values are in valid range [0, 1]

## If Something Doesn't Work

### SCOPE button is greyed out?
```bash
# Check if backend is running
lsof -i :5000  # on macOS/Linux
netstat -ano | findstr :5000  # on Windows

# If not, start it explicitly
cd turbine/scope/server && python -m flask --app app run --port 5000
```

### "Module not found" errors?
```bash
cd turbine/scope/server
pip install -r requirements.txt

# If still failing, install individually
pip install Flask==2.3.3
pip install flask-cors==4.0.0
pip install numpy==1.24.3
pip install scipy==1.11.2
```

### Frontend won't start?
```bash
cd hieronymus
npm install  # Reinstall dependencies
npm run dev
```

### CORS errors in browser console?
The backend has CORS enabled, but try:
1. Hard refresh the page (Ctrl+Shift+R or Cmd+Shift+R)
2. Clear browser cache
3. Check backend is running on port 5000

## What's Been Integrated

### ✅ Backend (Python/Flask)
- `turbine/scope/server/app.py` — HTTP API for SCOPE execution
- `/health` — Health check
- `/programs` — List available programs
- `/execute` — Execute SCOPE program with timing events + frame
- Program registry: `nuclear_separation_dynamics` (example)

### ✅ Frontend (TypeScript/React)
- `hieronymus/src/lib/scope-client.ts` — HTTP client for SCOPE API
- `hieronymus/src/app/tools/analysis-studio/page.tsx` — Updated UI with SCOPE mode
- Mode selector: Switch between JavaScript and SCOPE
- Program selector: Choose from available programs
- Phase selector: Select cell cycle phase (for nuclear_separation)
- Execution: Click Run to execute SCOPE program

### ✅ SCOPE Runtime
- Five phases: COMPILE → ASSIGN → MEASURE → EXECUTE → EMIT
- Timing event classification into temporal cells
- Spectral pipeline for coordinate field estimation
- Morphism chain execution with world-space grounding
- Formal uncertainty bounds and entropy tracking

## Next Steps After Verification

### Manual Testing
1. Try all three phases: PROPHASE, METAPHASE, ANAPHASE
2. Verify distance values change appropriately
3. Check S-entropy conservation (S_k + S_t + S_e = 1.0)

### Performance Testing
```bash
# Time a single execution
time curl -X POST http://localhost:5000/execute \
  -H "Content-Type: application/json" \
  -d '{"program_id": "nuclear_separation_dynamics", ...}'
```

### Browser Console
1. Open DevTools (F12)
2. Switch to SCOPE mode
3. Watch Network tab as requests are made
4. Response should be valid JSON with results

## File Structure

```
hieronymus/
├── src/
│   ├── lib/
│   │   └── scope-client.ts           ← TypeScript client library
│   └── app/tools/
│       └── analysis-studio/
│           └── page.tsx              ← Updated UI with SCOPE mode
│
turbine/
├── scope/
│   ├── server/
│   │   ├── app.py                   ← Flask backend
│   │   └── requirements.txt          ← Python dependencies
│   ├── types/
│   ├── phases/
│   ├── runtime/
│   └── programs/
│       └── nuclear_separation.py    ← Example program
```

## Documentation

- **Full Integration Guide**: `INTEGRATION_GUIDE.md`
- **SCOPE Implementation**: `SCOPE_IMPLEMENTATION_STATUS.md`
- **SCOPE Paper**: `hieronymus/publications/sources/scope-metalanguage.tex`
- **Python DSL Guide**: `turbine/scope/README.md`

---

**Questions?** Check `INTEGRATION_GUIDE.md` for detailed troubleshooting.

**Ready?** Start the backend and frontend, then test! 🚀
