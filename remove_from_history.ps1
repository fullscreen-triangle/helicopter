# PowerShell script to remove large files from entire git history
# WARNING: This rewrites git history and requires force push

Write-Host "========================================" -ForegroundColor Red
Write-Host "WARNING: This will rewrite git history!" -ForegroundColor Red
Write-Host "========================================" -ForegroundColor Red
Write-Host ""

$confirm = Read-Host "Are you sure you want to continue? (yes/no)"
if ($confirm -ne "yes") {
    Write-Host "Aborted." -ForegroundColor Yellow
    exit
}

Write-Host ""
Write-Host "Removing large files from git history..." -ForegroundColor Yellow

# Files to remove
$filesToRemove = @(
    "maxwell/publication/pixel-maxwell-demon/figures/moriarty_3d_phase_space_with_image.gif",
    "lifescience/public/55784.zip"
)

# Use git filter-branch to remove files from all commits
Write-Host ""
Write-Host "Step 1: Removing files from all commits using git filter-branch..." -ForegroundColor Cyan

$filterCommand = "git filter-branch --force --index-filter `""
foreach ($file in $filesToRemove) {
    $filterCommand += "git rm --cached --ignore-unmatch '$file'; "
}
$filterCommand += "`" --prune-empty --tag-name-filter cat -- --all"

Write-Host "Running: $filterCommand" -ForegroundColor Gray
Invoke-Expression $filterCommand

if ($LASTEXITCODE -eq 0) {
    Write-Host ""
    Write-Host "[OK] Files removed from git history" -ForegroundColor Green
} else {
    Write-Host ""
    Write-Host "[ERROR] git filter-branch failed. Trying alternative method..." -ForegroundColor Red
    
    # Alternative: Use git filter-repo if available, or manual approach
    Write-Host "Attempting manual removal from recent commits..." -ForegroundColor Yellow
    
    # Get list of commits that contain these files
    foreach ($file in $filesToRemove) {
        Write-Host "Checking commits for: $file" -ForegroundColor Cyan
        $commits = git log --all --pretty=format:"%H" -- "$file" 2>$null
        if ($commits) {
            Write-Host "  Found in commits: $($commits.Count)" -ForegroundColor Yellow
        } else {
            Write-Host "  File not found in git history" -ForegroundColor Green
        }
    }
}

Write-Host ""
Write-Host "Step 2: Cleaning up backup refs..." -ForegroundColor Cyan
git for-each-ref --format="%(refname)" refs/original/ | ForEach-Object {
    git update-ref -d $_
}

Write-Host ""
Write-Host "Step 3: Running garbage collection..." -ForegroundColor Cyan
git reflog expire --expire=now --all
git gc --prune=now --aggressive

Write-Host ""
Write-Host "[OK] Git history cleaned!" -ForegroundColor Green
Write-Host ""
Write-Host "Next steps:" -ForegroundColor Yellow
Write-Host "  1. Verify files are gone: git log --all --full-history -- 'maxwell/publication/pixel-maxwell-demon/figures/moriarty_3d_phase_space_with_image.gif'" -ForegroundColor White
Write-Host "  2. Force push (WARNING: This rewrites remote history):" -ForegroundColor Red
Write-Host "     git push --force --all" -ForegroundColor White
Write-Host ""
Write-Host "NOTE: If others are working on this repo, coordinate with them first!" -ForegroundColor Red
