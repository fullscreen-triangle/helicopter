@echo off
REM Create ArXiv submission ZIP file

echo ================================
echo Creating ArXiv Submission ZIP
echo ================================
echo.

REM Check if files exist
if not exist "lunar-surface-arxiv.tex" (
    echo ERROR: lunar-surface-arxiv.tex not found!
    pause
    exit /b 1
)

if not exist "references.bib" (
    echo ERROR: references.bib not found!
    pause
    exit /b 1
)

echo Checking for PNG figures...
set /a count=0
for %%f in (*.png) do set /a count+=1
echo Found %count% PNG files

if %count% LSS 13 (
    echo WARNING: Expected 13 PNG files, found %count%
    echo.
    echo Required figures:
    echo   - section_2_validation.png
    echo   - section_3_validation.png
    echo   - section_4_validation.png
    echo   - 3D_VOLUMETRIC_RECONSTRUCTION.png
    echo   - section_5_validation.png
    echo   - section_6_validation.png
    echo   - section_7_validation.png
    echo   - section_8_validation.png
    echo   - LUNAR_FEATURES_DEMONSTRATION.png
    echo   - section_9_validation.png
    echo   - LUNAR_DUST_DISPLACEMENT_ANALYSIS.png
    echo   - ECLIPSE_SHADOW_CALCULATION.png
    echo   - lunar_virtual_imaging_demonstration.png
    echo.
    pause
)

REM Create ZIP using PowerShell
echo.
echo Creating lunar-surface-submission.zip...
powershell -Command "Compress-Archive -Path lunar-surface-arxiv.tex,references.bib,*.png -DestinationPath lunar-surface-submission.zip -Force"

if exist "lunar-surface-submission.zip" (
    echo.
    echo ================================
    echo SUCCESS!
    echo ================================
    echo.
    echo ZIP file created: lunar-surface-submission.zip
    echo.
    for %%f in (lunar-surface-submission.zip) do echo File size: %%~zf bytes (~%%~zf KB)
    echo.
    echo Contents:
    powershell -Command "Get-Content lunar-surface-submission.zip | Select-Object -First 0; (New-Object -ComObject Shell.Application).NameSpace((Resolve-Path 'lunar-surface-submission.zip').Path).Items() | Select-Object Name, Size | Format-Table -AutoSize"
    echo.
    echo Next steps:
    echo 1. Upload lunar-surface-submission.zip to https://arxiv.org/submit
    echo 2. Select category: astro-ph.EP (Earth and Planetary Astrophysics)
    echo 3. Fill in metadata (title, author, abstract)
    echo 4. Submit!
    echo.
) else (
    echo.
    echo ERROR: Failed to create ZIP file
    pause
    exit /b 1
)

pause

