@echo off
REM Test arXiv compilation locally before submission

echo ================================
echo Testing arXiv LaTeX Compilation
echo ================================
echo.

echo [1/4] Running pdflatex (first pass)...
pdflatex -interaction=nonstopmode lunar-surface-arxiv.tex
if %errorlevel% neq 0 (
    echo ERROR: First pdflatex pass failed!
    echo Check the .log file for errors.
    pause
    exit /b 1
)

echo.
echo [2/4] Running bibtex...
bibtex lunar-surface-arxiv
if %errorlevel% neq 0 (
    echo ERROR: BibTeX failed!
    echo Check if references.bib exists.
    pause
    exit /b 1
)

echo.
echo [3/4] Running pdflatex (second pass)...
pdflatex -interaction=nonstopmode lunar-surface-arxiv.tex
if %errorlevel% neq 0 (
    echo ERROR: Second pdflatex pass failed!
    pause
    exit /b 1
)

echo.
echo [4/4] Running pdflatex (final pass)...
pdflatex -interaction=nonstopmode lunar-surface-arxiv.tex
if %errorlevel% neq 0 (
    echo ERROR: Final pdflatex pass failed!
    pause
    exit /b 1
)

echo.
echo ================================
echo SUCCESS! PDF compiled without errors.
echo ================================
echo.
echo Output file: lunar-surface-arxiv.pdf
echo.
echo The document should now compile on arXiv.
echo.
echo Next steps:
echo 1. Upload lunar-surface-arxiv.tex to arXiv
echo 2. Upload references.bib
echo 3. Upload all 13 PNG figures:
echo    - section_2_validation.png
echo    - section_3_validation.png
echo    - section_4_validation.png
echo    - 3D_VOLUMETRIC_RECONSTRUCTION.png
echo    - section_5_validation.png
echo    - section_6_validation.png
echo    - section_7_validation.png
echo    - section_8_validation.png
echo    - LUNAR_FEATURES_DEMONSTRATION.png
echo    - section_9_validation.png
echo    - LUNAR_DUST_DISPLACEMENT_ANALYSIS.png
echo    - ECLIPSE_SHADOW_CALCULATION.png
echo    - lunar_virtual_imaging_demonstration.png
echo.
pause

