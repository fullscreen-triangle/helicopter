# Git Repository Cleanup Guide

## Problem

GitHub rejects files larger than 20MB, and you have result files (videos, images, data) that exceed this limit:
- ❌ `*.mp4` - Video files
- ❌ `*.gif` - Animated images
- ❌ `*.csv` - Large data files
- ❌ `*.json` - JSON data files
- ❌ `*.npy` - NumPy arrays
- ❌ `*.png` - Large validation images (1000s of pixels)
- ❌ `*.pdf` - Generated PDFs

## Solution

### Step 1: Updated `.gitignore`

✅ **Added comprehensive exclusions**:

```gitignore
# Result directories
results/
**/results/
output/
**/output/
lunar_paper_validation/

# Large data files
*.csv
*.json
*.npy
*.npz
*.pkl
*.h5

# Videos and animations
*.mp4
*.gif
*.avi
*.mov

# Large validation images
*validation*.png
*demonstration*.png
*reconstruction*.png
*analysis*.png

# LaTeX outputs
*.pdf
*.aux
*.bbl
*.log
*.out

# Specific exclusions
pixel_maxwell_demon/lunar_paper_validation/**/*.png
pixel_maxwell_demon/docs/lunar-surface/*.pdf
```

### Step 2: Remove Files from Git Cache

Files in `.gitignore` won't be tracked for **new** files, but **existing** tracked files need to be removed from git's cache.

**Run the cleanup script**:
```cmd
clean_git_cache.bat
```

**Or manually**:
```bash
# Remove specific directories
git rm --cached -r pixel_maxwell_demon/lunar_paper_validation/
git rm --cached -r results/
git rm --cached -r output/

# Remove file types
git rm --cached "**/*.mp4"
git rm --cached "**/*.gif"
git rm --cached "**/*.csv"
git rm --cached "**/*.npy"
git rm --cached "**/*.pdf"

# Remove large PNGs (validation results)
git rm --cached "**/*validation*.png"
git rm --cached "**/*demonstration*.png"
git rm --cached "**/*reconstruction*.png"
```

### Step 3: Commit and Push

```bash
# Stage the .gitignore changes
git add .gitignore

# Commit the removals
git commit -m "Remove large files from tracking and update gitignore

- Exclude video files (*.mp4, *.gif)
- Exclude data files (*.csv, *.json, *.npy)
- Exclude validation outputs (lunar_paper_validation/)
- Exclude generated PDFs and LaTeX artifacts
- Files remain on disk but won't be tracked by git"

# Push to GitHub
git push
```

### Step 4: Clean Up Git History (Optional)

If files were previously committed and you want to reduce repository size:

```bash
# Aggressive garbage collection
git gc --aggressive --prune=now

# Force push (only if working alone!)
# git push --force
```

## File Categories

### ✅ Keep (Small, Essential)
- Source code (`.py`, `.js`, `.tex`)
- Configuration files (`package.json`, `.yaml`)
- Documentation (`.md` files)
- Small reference images (< 100 KB)
- BibTeX bibliography (`.bib`)

### ❌ Exclude (Large, Generated)
- **Videos**: All `*.mp4`, `*.gif`, `*.avi` files
- **Data**: `*.csv`, `*.json`, `*.npy`, `*.npz`
- **Results**: Validation outputs, experiments
- **Generated**: PDFs, large PNGs
- **Cache**: `__pycache__`, `.pytest_cache`
- **Build**: LaTeX aux files, compiled outputs

## Verify Before Pushing

```bash
# Check what's staged
git status

# Check file sizes
git ls-files | xargs ls -lh | sort -k5 -hr | head -20

# Check for files > 20MB
find . -type f -size +20M

# Dry run to see what would be pushed
git push --dry-run
```

## If Push Still Fails

### Option 1: Find Remaining Large Files
```powershell
# Find files > 20MB
Get-ChildItem -Recurse | Where-Object {$_.Length -gt 20MB} | Select-Object FullName, @{Name="Size(MB)";Expression={[math]::Round($_.Length/1MB, 2)}}
```

### Option 2: Use Git LFS (Large File Storage)
```bash
# Install Git LFS
git lfs install

# Track large file types
git lfs track "*.mp4"
git lfs track "*.gif"
git lfs track "*.csv"
git lfs track "*.npy"
git lfs track "*.pdf"

# Add .gitattributes
git add .gitattributes
git commit -m "Add Git LFS tracking"
```

### Option 3: Create Separate Data Repository
- Keep code in main repository
- Store large files in separate "data" repository
- Use submodules or references

## Common Issues

### Issue: "remote rejected... file too large"
**Cause**: File was previously committed and is in git history

**Solution**:
```bash
# Find large files in history
git rev-list --objects --all | git cat-file --batch-check='%(objecttype) %(objectname) %(objectsize) %(rest)' | grep '^blob' | sort -k3 -nr | head -20

# Remove specific file from history (DESTRUCTIVE!)
git filter-branch --force --index-filter "git rm --cached --ignore-unmatch PATH/TO/FILE" --prune-empty --tag-name-filter cat -- --all
```

### Issue: ".gitignore not working"
**Cause**: Files already tracked before .gitignore was updated

**Solution**:
```bash
# Remove from cache
git rm -r --cached .
git add .
git commit -m "Apply .gitignore rules"
```

## Best Practices

1. **Never commit**:
   - Model weights/checkpoints
   - Large datasets
   - Video files
   - Result outputs
   - Generated PDFs

2. **Use README to document**:
   - Where to get data
   - How to generate results
   - Links to external storage (Google Drive, Zenodo)

3. **Keep repository lean**:
   - < 1 GB total
   - < 100 MB per file
   - Only essential files

4. **For papers/publications**:
   - Commit LaTeX source (`.tex`, `.bib`)
   - Exclude PDFs (generated)
   - Exclude figures (generated from scripts)
   - Include figure generation scripts

## Summary

✅ **Updated `.gitignore`** - Comprehensive exclusions added  
✅ **Created `clean_git_cache.bat`** - Automated cleanup script  
✅ **Documented process** - This guide  

**Next Steps**:
1. Run `clean_git_cache.bat`
2. Review changes with `git status`
3. Commit: `git commit -m "Remove large files"`
4. Push: `git push`

**Result**: Clean repository under GitHub's limits! 🚀

