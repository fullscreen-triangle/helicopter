# SCOPE Compiler Specification

## Overview

The SCOPE compiler translates microscopy observation programs into executable five-phase pipelines. It has four stages: lexer, parser, type checker, code generator.

**Full pipeline:**
```
SCOPE Source → Lexer → Tokens → Parser → AST → Type Checker → Code Generator → ExecutionPlan
```

The executor then runs the `ExecutionPlan` with real image data through five phases:
1. **COMPILE** - accumulate timing events
2. **ASSIGN** - dispatch to morphism  
3. **MEASURE** - spectral pipeline → coordinate field Φ
4. **EXECUTE** - run morphism with Φ for world-space measurements
5. **EMIT** - return ObservationResult with S-entropy conservation check

---

## Lexer: Tokenization

Converts source code string to token stream. Recognizes:
- Keywords: `scope`, `channels`, `sync`, `cell`, `bounds`, `action`, `coordinate_space`, `field`, `depth`, `lambda_s`, `lambda_t`, `morphisms`, `observe`, `catalyze`, `fuse`, `measure_distance`, `access`, `dispatch`, `when`, `execute`, `emit`, `do`, `at`
- Symbols: `{`, `}`, `(`, `)`, `[`, `]`, `,`, `=`, `|>`, `:`, `µm`, `Hz`
- Literals: identifiers, numbers (integer and floating-point with scientific notation)
- Comments: `//` single-line and `/* ... */` multi-line

**No parsing complexity:** Straight regex-based tokenization, one pass.

---

## Parser: Syntax Tree

Recursive descent parser produces AST. Grammar (simplified BNF):

```
program    ::= 'scope' IDENT '{' channels coord_space morphisms dispatch '}'
channels   ::= 'channels' '{' (sync_decl | cell_decl)* '}'
sync_decl  ::= 'sync' IDENT 'at' NUMBER UNIT
cell_decl  ::= 'cell' IDENT 'bounds' '(' NUMBER ',' NUMBER ')' 'action' IDENT
coord_space ::= 'coordinate_space' '{' field_spec depth_spec coherence_spec '}'
field_spec ::= 'field' NUMBER 'x' NUMBER 'µm'
depth_spec ::= 'depth' INTEGER
coherence_spec ::= 'lambda_s' NUMBER 'lambda_t' NUMBER
morphisms  ::= 'morphisms' '{' chain_def* '}'
chain_def  ::= IDENT '=' observe_step ('|>' step)*
observe_step ::= 'observe' '(' IDENT ',' 'n' '=' INTEGER ')'
step       ::= catalyze | fuse | measure | access
catalyze   ::= 'catalyze' '(' IDENT ')'
fuse       ::= 'fuse' '(' IDENT ',' 'rho' '=' NUMBER ')'
measure    ::= 'measure_distance' '(' IDENT ',' IDENT ')'
access     ::= 'access' '(' IDENT ')'
dispatch   ::= 'dispatch' '{' when_rule* '}'
when_rule  ::= 'when' IDENT 'do' 'execute' '(' IDENT ')'
```

Produces AST nodes for each syntactic element. Fails immediately on parse error with line/column.

---

## Type Checker: Constraint Verification

Verifies four invariants:

### 1. Partition Depth Consistency
**Rule**: All `observe(frame, n=VALUE)` steps must use the same depth `n` as declared in `coordinate_space { depth N }`.

**Failure**: Error with location
```
Morphism 'measure_nuclei': observe depth 2000 does not match coordinate_space depth 1000
```

**Why**: The depth defines the spatial resolution. If mismatched, coordinate field Φ generation (Phase 3) and morphism execution (Phase 4) work at different scales.

### 2. S-Entropy Conservation Heuristic
**Rule**: Each morphism chain's catalyze/access steps should be roughly balanced. Heuristic: if `|catalyze_count - access_count| > 3`, warn.

**Failure**: Warning only (not fatal)
```
Morphism 'measure_nuclei': catalyze 5 vs access 1 steps seem imbalanced (S-entropy may not conserve)
```

**Why**: Catalyze reduces S_k, access increases S_e. Extreme imbalance suggests the chain won't conserve S_k + S_t + S_e = 1.

### 3. Dispatch Completeness
**Rule**: Every cell in `channels { cell CELL_ID ... }` must have a `when CELL_ID do execute(...)` rule in `dispatch`.

**Failure**: Error
```
Cell 'PROPHASE' has no dispatch rule
```

**Why**: Incomplete dispatch means some timing cells won't trigger observations.

### 4. Morphism Existence
**Rule**: Every action reference (in cell `action` or dispatch `do execute(...)`) must reference an existing morphism name.

**Failure**: Error
```
Cell 'PROPHASE': action 'measure_nuclei' is not defined in morphisms
```

**Why**: Broken references cause runtime crashes.

---

## Code Generator: Execution Plan

Emits `ExecutionPlan` JSON with structure:

```typescript
{
  "name": "nuclear_separation_prophase",
  "coordinate_space": {
    "field_width_um": 100.0,
    "field_height_um": 100.0,
    "depth": 1000,
    "lambda_s": 0.10,
    "lambda_t": 0.05
  },
  "channels": {
    "sync": {
      "id": "acquisition",
      "frequency": 10000000  // 10 MHz
    },
    "cells": [
      {
        "id": "PROPHASE",
        "bounds_min": -2.0e-6,
        "bounds_max": -0.8e-6,
        "morphism_id": "measure_nuclei"
      },
      {
        "id": "METAPHASE",
        "bounds_min": -0.8e-6,
        "bounds_max": 0.8e-6,
        "morphism_id": "measure_nuclei"
      }
    ]
  },
  "morphisms": [
    {
      "id": "measure_nuclei",
      "steps": [
        {
          "type": "observe",
          "params": {
            "frame": "dapi_frame",
            "depth": 1000
          }
        },
        {
          "type": "catalyze",
          "params": {
            "constraint": "conservation(dna_mass)"
          }
        },
        {
          "type": "measure",
          "params": {
            "target_a": "nucleus_a",
            "target_b": "nucleus_b"
          }
        },
        {
          "type": "access",
          "params": {
            "structure": "separation_vector"
          }
        }
      ]
    }
  ]
}
```

The executor reads this plan and orchestrates the five-phase pipeline.

---

## Implementation Checklist

- [ ] `src/scope-compiler/lexer.ts` — Tokenize SCOPE source
- [ ] `src/scope-compiler/parser.ts` — Build AST
- [ ] `src/scope-compiler/type-checker.ts` — Verify invariants
- [ ] `src/scope-compiler/code-generator.ts` — Emit ExecutionPlan
- [ ] `src/scope-compiler/index.ts` — Main entry: `compileScope(sourceCode) → ExecutionPlan`
- [ ] Tests for each stage (lexer, parser, type checker)
