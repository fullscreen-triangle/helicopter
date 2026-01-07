@echo off
REM Clean large files from git cache and prepare for push

echo ====================================
echo Git Repository Cleanup Script
echo ====================================
echo.

echo Step 1: Checking for large files currently tracked...
echo.
git ls-files | findstr /i "\.mp4$ \.gif$ \.csv$ \.json$ \.npy$ \.npz$ \.pkl$ \.pdf$ \.png$" > large_files_list.txt

echo Found potentially large files:
type large_files_list.txt
echo.

echo Step 2: Removing large files from git cache...
echo (Files will remain on disk, just untracked)
echo.

REM Remove validation outputs
git rm --cached -r pixel_maxwell_demon/lunar_paper_validation/ 2>NUL
git rm --cached -r pixel_maxwell_demon/multi_modal_validation/ 2>NUL
git rm --cached -r pixel_maxwell_demon/spectral_multiplexing_validation/ 2>NUL
git rm --cached -r pixel_maxwell_demon/motion_picture_validation/ 2>NUL

REM Remove result directories
git rm --cached -r results/ 2>NUL
git rm --cached -r output/ 2>NUL
git rm --cached -r experiments/ 2>NUL

REM Remove large file types
echo Removing *.mp4 files...
for /f "tokens=*" %%f in ('git ls-files ^| findstr /i "\.mp4$"') do (
    git rm --cached "%%f" 2>NUL
)

echo Removing *.gif files...
for /f "tokens=*" %%f in ('git ls-files ^| findstr /i "\.gif$"') do (
    git rm --cached "%%f" 2>NUL
)

echo Removing *.csv files...
for /f "tokens=*" %%f in ('git ls-files ^| findstr /i "\.csv$"') do (
    git rm --cached "%%f" 2>NUL
)

echo Removing *.json files (except package.json, tsconfig.json, etc.)...
for /f "tokens=*" %%f in ('git ls-files ^| findstr /i "\.json$" ^| findstr /v "package.json tsconfig.json"') do (
    git rm --cached "%%f" 2>NUL
)

echo Removing *.npy files...
for /f "tokens=*" %%f in ('git ls-files ^| findstr /i "\.npy$"') do (
    git rm --cached "%%f" 2>NUL
)

echo Removing *.npz files...
for /f "tokens=*" %%f in ('git ls-files ^| findstr /i "\.npz$"') do (
    git rm --cached "%%f" 2>NUL
)

echo Removing *.pkl files...
for /f "tokens=*" %%f in ('git ls-files ^| findstr /i "\.pkl$"') do (
    git rm --cached "%%f" 2>NUL
)

echo Removing large *.png files (validation results)...
for /f "tokens=*" %%f in ('git ls-files ^| findstr /i "validation.*\.png$ demonstration.*\.png$ reconstruction.*\.png$ analysis.*\.png$"') do (
    git rm --cached "%%f" 2>NUL
)

echo Removing PDF files...
for /f "tokens=*" %%f in ('git ls-files ^| findstr /i "\.pdf$"') do (
    git rm --cached "%%f" 2>NUL
)

echo Removing LaTeX aux files...
for /f "tokens=*" %%f in ('git ls-files ^| findstr /i "\.aux$ \.bbl$ \.blg$ \.log$ \.out$ \.toc$"') do (
    git rm --cached "%%f" 2>NUL
)

echo.
echo Step 3: Checking repository size...
echo.
git count-objects -vH

echo.
echo Step 4: Clean up (optional but recommended)
echo.
echo To reduce repository size further, run:
echo   git gc --aggressive --prune=now
echo.
choice /C YN /M "Run git gc now?"
if errorlevel 2 goto skip_gc
if errorlevel 1 goto run_gc

:run_gc
echo Running git gc...
git gc --aggressive --prune=now
goto done_gc

:skip_gc
echo Skipping git gc

:done_gc
echo.
echo ====================================
echo Cleanup Complete!
echo ====================================
echo.
echo Next steps:
echo 1. Review changes: git status
echo 2. Commit the removals: git add .gitignore
echo 3. Commit: git commit -m "Remove large files from tracking"
echo 4. Push: git push
echo.
echo Note: Large files remain on your disk but won't be tracked by git
echo.

del large_files_list.txt 2>NUL

pause

