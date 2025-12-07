@echo off
REM Compilation script for Virtual Imaging via Dual-Membrane Pixel Maxwell Demons
REM Windows batch file

echo ================================================================================
echo   COMPILING: Virtual Imaging via Dual-Membrane Pixel Maxwell Demons
echo ================================================================================
echo.

REM Check if pdflatex is available
where pdflatex >nul 2>nul
if %ERRORLEVEL% NEQ 0 (
    echo ERROR: pdflatex not found. Please install MiKTeX or TeX Live.
    echo   Download from: https://miktex.org/ or https://www.tug.org/texlive/
    exit /b 1
)

set MAIN_TEX=virtual-imaging-membrane-pixels

echo Step 1/4: First LaTeX compilation...
pdflatex -interaction=nonstopmode "%MAIN_TEX%.tex" >nul 2>&1
if %ERRORLEVEL% EQU 0 (
    echo   [OK] First compilation successful
) else (
    echo   [ERROR] First compilation failed. Running with output:
    pdflatex "%MAIN_TEX%.tex"
    exit /b 1
)

echo Step 2/4: BibTeX compilation...
bibtex "%MAIN_TEX%" >nul 2>&1
if %ERRORLEVEL% EQU 0 (
    echo   [OK] BibTeX successful
) else (
    echo   [ERROR] BibTeX failed. Running with output:
    bibtex "%MAIN_TEX%"
    exit /b 1
)

echo Step 3/4: Second LaTeX compilation...
pdflatex -interaction=nonstopmode "%MAIN_TEX%.tex" >nul 2>&1
if %ERRORLEVEL% EQU 0 (
    echo   [OK] Second compilation successful
) else (
    echo   [ERROR] Second compilation failed
    exit /b 1
)

echo Step 4/4: Final LaTeX compilation...
pdflatex -interaction=nonstopmode "%MAIN_TEX%.tex" >nul 2>&1
if %ERRORLEVEL% EQU 0 (
    echo   [OK] Final compilation successful
) else (
    echo   [ERROR] Final compilation failed
    exit /b 1
)

echo.
echo ================================================================================
echo   COMPILATION COMPLETE
echo ================================================================================
echo.
echo Output: %MAIN_TEX%.pdf
echo.

if exist "%MAIN_TEX%.pdf" (
    for %%A in ("%MAIN_TEX%.pdf") do echo PDF size: %%~zA bytes
    echo.
    echo To view the PDF:
    echo   start %MAIN_TEX%.pdf
)

echo.
echo ================================================================================

pause

