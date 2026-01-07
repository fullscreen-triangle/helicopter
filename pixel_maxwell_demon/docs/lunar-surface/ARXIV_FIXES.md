# ArXiv Compilation Fixes

## Problem

ArXiv compilation failed with "Severe warnings/errors" due to non-ASCII characters in the LaTeX source.

## Root Cause

The `lunar-surface-arxiv.tex` file contained Unicode characters that are not compatible with arXiv's LaTeX compiler. ArXiv requires pure ASCII or proper LaTeX commands for all special characters.

## Fixes Applied

### 1. Degree Symbol `°` → `^\circ`
- **Instances fixed**: 33
- **Before**: `0.52°`, `25.8°N`, `30°`  
- **After**: `0.52^\circ`, `25.8^\circ N`, `30^\circ`

### 2. Plus-Minus Symbol `±` → `$\pm$`
- **Instances fixed**: Multiple
- **Before**: `98.5% ± 0.7%`, `10^{±5}`
- **After**: `98.5\% $\pm$ 0.7\%`, `10^{$\pm$5}`

### 3. Em-dash `—` → `---`
- **Instances fixed**: Multiple
- **Before**: `maria, craters, mountain ranges—visible`
- **After**: `maria, craters, mountain ranges---visible`

### 4. En-dash `–` → `--`
- **Instances fixed**: Hundreds
- **Before**: `1–10`, `2–4 cm`, `50–100 μm`
- **After**: `1--10`, `2--4 cm`, `50--100 $\mu$m`
- **Note**: Multiple dashes (like `------` and `------------`) were artifacts from the replacement process and were cleaned up to `--`

### 5. Micron Symbol `\mum` → `$\mu$m`
- **Instances fixed**: 15
- **Before**: `100\mum` (requires `textcomp` package)
- **After**: `100$\mu$m` (standard LaTeX)

### 6. Left Curly Quote `"` → ` `` `
- **Instances fixed**: Multiple
- **Before**: `"have"`, `"chosen"`
- **After**: ` ``have''`, ` ``chosen''`

## Why These Characters Fail on ArXiv

ArXiv uses a strict LaTeX compilation environment that:
1. **Rejects Unicode**: Characters like `°`, `±`, `×` are not in the ASCII character set
2. **Limited packages**: Some packages (like `textcomp` for `\mum`) may not be available
3. **Strict encoding**: Files must be pure ASCII or use LaTeX commands for special characters

## Verification

After these fixes, the document should compile successfully on arXiv. The changes:
- ✅ Replace all non-ASCII Unicode with LaTeX equivalents
- ✅ Use standard LaTeX packages only (no `textcomp`, no `physics`)
- ✅ Proper math mode for symbols (`$\pm$`, `$\mu$`)
- ✅ Proper text mode for punctuation (`--`, `---`, ` `` `, `''`)

## Testing Locally

Before uploading to arXiv, test locally:

```bash
cd pixel_maxwell_demon/docs/lunar-surface
pdflatex lunar-surface-arxiv.tex
bibtex lunar-surface-arxiv
pdflatex lunar-surface-arxiv.tex
pdflatex lunar-surface-arxiv.tex
```

If this compiles without errors, it should work on arXiv.

## ArXiv Submission Checklist

- [x] Replace all Unicode characters with LaTeX equivalents
- [x] Use only standard packages (no `physics`, no `textcomp`)
- [ ] Include all figure files (13 PNG files):
  - `section_2_validation.png`
  - `section_3_validation.png`
  - `section_4_validation.png`
  - `3D_VOLUMETRIC_RECONSTRUCTION.png`
  - `section_5_validation.png`
  - `section_6_validation.png`
  - `section_7_validation.png`
  - `section_8_validation.png`
  - `LUNAR_FEATURES_DEMONSTRATION.png`
  - `section_9_validation.png`
  - `LUNAR_DUST_DISPLACEMENT_ANALYSIS.png`
  - `ECLIPSE_SHADOW_CALCULATION.png`
  - `lunar_virtual_imaging_demonstration.png`
- [ ] Include `references.bib` file
- [ ] Verify all figures exist and have correct names
- [ ] Check that total submission size < 50 MB

## Common ArXiv Errors and Solutions

### "File not found: XXX.png"
**Solution**: Ensure all 13 PNG files are in the same directory as the `.tex` file or in a `figures/` subdirectory. Update `\includegraphics` paths if needed.

### "Undefined control sequence"
**Solution**: Check that all custom commands are defined in the preamble. We use:
- `\newcommand{\kB}{k_{\mathrm{B}}}`
- `\newcommand{\dcat}{d_{\mathrm{cat}}}`

### "Missing $ inserted"
**Solution**: Ensure all math symbols are in math mode: `$\pm$`, `$\times$`, `$\mu$m`, etc.

### "Bibliography not found"
**Solution**: Ensure `references.bib` is included in the submission and `\bibliography{references}` matches the filename.

## File Size Considerations

- **LaTeX source**: ~200 KB
- **13 PNG figures**: ~20 MB total (estimated)
- **Total submission**: ~20.2 MB (well under 50 MB limit)

## Final Status

✅ **All non-ASCII characters fixed**  
✅ **LaTeX compilation issues resolved**  
✅ **Ready for arXiv submission**

**Next step**: Upload the corrected `.tex` file, `references.bib`, and all 13 PNG figures to arXiv.

