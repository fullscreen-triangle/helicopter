# SCOPE TypeScript Compiler

**Date**: 2026-05-28  
**Status**: ✅ Complete and integrated into Analysis Studio

## What Was Built

A complete **TypeScript-based SCOPE compiler** that runs entirely in the browser (frontend-only). No backend required for compilation.

## Architecture

### 1. Lexer (`lexer.ts`)
- Tokenizes SCOPE source code
- Recognizes all SCOPE keywords, symbols, and literals
- Handles strings, numbers, identifiers
- Error reporting with line/column information

**Key token types:**
- Keywords: `scope`, `channels`, `sync`, `cell`, `coordinate_space`, `morphisms`, `dispatch`, etc.
- Symbols: `{`, `}`, `(`, `)`, `|>` (pipe), `=`, etc.
- Literals: identifiers, numbers, strings

### 2. Parser (`parser.ts`)
- Builds Abstract Syntax Tree (AST) from tokens
- Implements full SCOPE grammar (BNF)
- Semantic structure:

```
ScopeProgram
├── ChannelsBlock (sync declarations + cell declarations)
├── CoordinateSpaceBlock (field size, depth, lambda parameters)
├── MorphismsBlock (chains of analysis steps)
└── DispatchBlock (when-do rules mapping cells to actions)
```

**Morphism steps:** observe, catalyze, fuse, measure_distance, access

### 3. Compiler (`compiler.ts`)
- Generates Intermediate Representation (IR) from AST
- Performs semantic validation:
  - Check all referenced chains exist
  - Check all referenced cells exist
  - Warn about unused chains
- Outputs structured IR ready for execution

**IR Includes:**
- Channel definitions (sync/cell)
- Coordinate space parameters
- Morphism chains with step parameters
- Dispatch table mapping cells to actions

### 4. Integration (`index.ts`)
- Exports lexer, parser, compiler
- Single entry point: `compileScope(source: string) -> CompiledProgram`

## Updated Analysis Studio

**File**: `hieronymus/src/app/tools/analysis-studio/page.tsx`

**Features:**
1. **SCOPE-first UI** — removed JavaScript mode entirely
2. **Live compiler** — compiles on "Compile & Run" button
3. **Error reporting** — displays compilation errors with context
4. **Program inspection** — shows compiled structure (channels, morphisms, dispatch rules)
5. **Image selection** — database browser to choose real or synthetic images
6. **Console output** — real-time compilation messages

**Sample SCOPE program included:**
- `nuclear_separation_dynamics` — demonstrates all SCOPE features
- 3 cell cycle phases (PROPHASE, METAPHASE, ANAPHASE)
- Morphism chain with observe → catalyze → measure_distance → access steps
- Dispatch rules mapping phases to actions

## How to Use

1. **Open Analysis Studio**: `http://localhost:3007/tools/analysis-studio` (or current port)
2. **Edit SCOPE program**: Write or modify SCOPE code in the left panel
3. **Click "Compile & Run"**: Compiles the program
4. **View results**:
   - Success: Shows compiled program structure
   - Errors: Lists compilation errors with line numbers
   - Warnings: Shows unused morphisms
5. **Select image** (optional): Use database browser to choose real BBBC image

## Example SCOPE Program

```scope
scope nuclear_separation_dynamics {

    channels {
        sync acquisition at 10.0e6
        cell PROPHASE   bounds (-2.0e-6, -0.8e-6)
        cell METAPHASE  bounds (-0.8e-6,  0.8e-6)
        cell ANAPHASE   bounds ( 0.8e-6,  2.0e-6)
    }

    coordinate_space {
        field   100.0 x 100.0 µm
        depth   1000
        lambda_s  0.10
        lambda_t  0.05
    }

    morphisms {
        nucleus_pair_measurement =
            observe(frame_t, n=1000)
            |> catalyze(conservation)
            |> measure_distance(nucleus_a, nucleus_b)
            |> access(separation_vector)
    }

    dispatch {
        when PROPHASE  do execute(nucleus_pair_measurement)
        when METAPHASE do execute(nucleus_pair_measurement)
        when ANAPHASE  do execute(nucleus_pair_measurement)
    }
}
```

## Files Created

```
hieronymus/src/lib/scope-compiler/
├── lexer.ts          (302 lines) — tokenization
├── parser.ts         (428 lines) — AST generation
├── compiler.ts       (230 lines) — IR generation + validation
└── index.ts          (8 lines)  — module exports

hieronymus/src/app/tools/analysis-studio/
└── page.tsx          (348 lines) — refactored to use SCOPE compiler
```

## Next Steps

### Immediate (Optional)
- Create TypeScript SCOPE runtime to execute compiled programs
- Add visualization of compiled IR structure
- Add step-by-step debugging of morphism chains

### Medium-term
- Build Rust SCOPE runtime (higher performance)
- Port spectral pipeline (measure_distance) to Rust
- Connect TypeScript compiler → Rust runtime via WebAssembly/WASM

### Long-term
- GPU acceleration for spectral analysis
- Time-lapse sequence support
- 3D volume support
- Result archival and comparison

## Key Design Decisions

1. **Frontend-only compilation** — no backend needed for syntactic validation
2. **AST + IR separation** — AST for error reporting, IR for execution
3. **Strict validation** — all references checked before execution
4. **No JavaScript fallback** — SCOPE is the only analysis language
5. **Database integration** — real image selection built-in

## Current Limitations

- IR is generated but not executed (execution requires runtime)
- No actual morphism execution yet (awaiting Rust runtime)
- No image processing (awaiting full runtime implementation)
- Sample program is structural only (not functionally evaluated)

## Testing

The compiler has been integrated into the Analysis Studio and is ready for testing:

1. Syntax validation — try breaking SCOPE syntax, see errors
2. Semantic validation — reference undefined chains, see warnings
3. Program inspection — examine compiled program structure
4. Real images — select from BBBC database (ready for future runtime)

The TypeScript compiler fulfills the requirement for a **frontend-only SCOPE implementation** without any Python or backend dependencies.
