# PowerShell script to remove large GIF, video, and ZIP files from git tracking
# Run this script to unstage large files that exceed GitHub's 100MB limit

Write-Host "Removing large files from git tracking..." -ForegroundColor Yellow

# Remove the specific large files mentioned in the error
Write-Host ""
Write-Host "Removing large GIF file..." -ForegroundColor Cyan
git rm --cached "maxwell/publication/pixel-maxwell-demon/figures/moriarty_3d_phase_space_with_image.gif" 2>$null
if ($LASTEXITCODE -eq 0) {
    Write-Host "  [OK] Removed GIF file from git tracking" -ForegroundColor Green
} else {
    Write-Host "  [WARN] File may not be tracked or already removed" -ForegroundColor Yellow
}

Write-Host ""
Write-Host "Removing large ZIP file..." -ForegroundColor Cyan
git rm --cached "lifescience/public/55784.zip" 2>$null
if ($LASTEXITCODE -eq 0) {
    Write-Host "  [OK] Removed ZIP file from git tracking" -ForegroundColor Green
} else {
    Write-Host "  [WARN] File may not be tracked or already removed" -ForegroundColor Yellow
}

# Remove all GIF files from git tracking
Write-Host ""
Write-Host "Removing all GIF files from git tracking..." -ForegroundColor Cyan
$gifFiles = git ls-files | Where-Object { $_ -match '\.gif$' -or $_ -match '\.GIF$' }
foreach ($file in $gifFiles) {
    git rm --cached $file 2>$null
    if ($LASTEXITCODE -eq 0) {
        Write-Host "  [OK] Removed: $file" -ForegroundColor Green
    }
}

# Remove all video files from git tracking
Write-Host ""
Write-Host "Removing all video files from git tracking..." -ForegroundColor Cyan
$videoFiles = git ls-files | Where-Object { 
    $_ -match '\.mp4$' -or $_ -match '\.avi$' -or $_ -match '\.mov$' -or 
    $_ -match '\.mkv$' -or $_ -match '\.webm$' -or $_ -match '\.flv$' -or
    $_ -match '\.MP4$' -or $_ -match '\.AVI$' -or $_ -match '\.MOV$' -or $_ -match '\.MKV$'
}
foreach ($file in $videoFiles) {
    git rm --cached $file 2>$null
    if ($LASTEXITCODE -eq 0) {
        Write-Host "  [OK] Removed: $file" -ForegroundColor Green
    }
}

# Remove large ZIP files from git tracking
Write-Host ""
Write-Host "Removing large ZIP files from git tracking..." -ForegroundColor Cyan
$zipFiles = git ls-files | Where-Object { $_ -match '\.zip$' -or $_ -match '\.ZIP$' }
foreach ($file in $zipFiles) {
    if (Test-Path $file) {
        $fileSize = (Get-Item $file).Length / 1MB
        if ($fileSize -gt 10) {  # Only remove ZIPs larger than 10MB
            git rm --cached $file 2>$null
            if ($LASTEXITCODE -eq 0) {
                Write-Host "  [OK] Removed: $file ($([math]::Round($fileSize, 2)) MB)" -ForegroundColor Green
            }
        }
    }
}

Write-Host ""
Write-Host "[OK] Done! Large files removed from git tracking." -ForegroundColor Green
Write-Host ""
Write-Host "Next steps:" -ForegroundColor Yellow
Write-Host "  1. Review changes: git status" -ForegroundColor White
Write-Host "  2. Commit the removal: git commit -m 'Remove large GIF/video/ZIP files from tracking'" -ForegroundColor White
Write-Host "  3. Push: git push" -ForegroundColor White
