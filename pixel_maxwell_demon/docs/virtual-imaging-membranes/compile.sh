#!/bin/bash

# Compilation script for Virtual Imaging via Dual-Membrane Pixel Maxwell Demons

echo "================================================================================"
echo "  COMPILING: Virtual Imaging via Dual-Membrane Pixel Maxwell Demons"
echo "================================================================================"
echo ""

# Check if pdflatex is available
if ! command -v pdflatex &> /dev/null; then
    echo "ERROR: pdflatex not found. Please install a LaTeX distribution."
    echo "  - Linux: sudo apt-get install texlive-full"
    echo "  - macOS: brew install --cask mactex"
    echo "  - Windows: Install MiKTeX or TeX Live"
    exit 1
fi

# Check if bibtex is available
if ! command -v bibtex &> /dev/null; then
    echo "ERROR: bibtex not found. Please install a LaTeX distribution."
    exit 1
fi

MAIN_TEX="virtual-imaging-membrane-pixels"

echo "Step 1/4: First LaTeX compilation..."
pdflatex -interaction=nonstopmode "$MAIN_TEX.tex" > /dev/null 2>&1
if [ $? -eq 0 ]; then
    echo "  ✓ First compilation successful"
else
    echo "  ✗ First compilation failed. Running with output:"
    pdflatex "$MAIN_TEX.tex"
    exit 1
fi

echo "Step 2/4: BibTeX compilation..."
bibtex "$MAIN_TEX" > /dev/null 2>&1
if [ $? -eq 0 ]; then
    echo "  ✓ BibTeX successful"
else
    echo "  ✗ BibTeX failed. Running with output:"
    bibtex "$MAIN_TEX"
    exit 1
fi

echo "Step 3/4: Second LaTeX compilation..."
pdflatex -interaction=nonstopmode "$MAIN_TEX.tex" > /dev/null 2>&1
if [ $? -eq 0 ]; then
    echo "  ✓ Second compilation successful"
else
    echo "  ✗ Second compilation failed"
    exit 1
fi

echo "Step 4/4: Final LaTeX compilation..."
pdflatex -interaction=nonstopmode "$MAIN_TEX.tex" > /dev/null 2>&1
if [ $? -eq 0 ]; then
    echo "  ✓ Final compilation successful"
else
    echo "  ✗ Final compilation failed"
    exit 1
fi

echo ""
echo "================================================================================"
echo "  COMPILATION COMPLETE"
echo "================================================================================"
echo ""
echo "Output: $MAIN_TEX.pdf"
echo ""

# Count pages
if command -v pdfinfo &> /dev/null; then
    PAGES=$(pdfinfo "$MAIN_TEX.pdf" 2>/dev/null | grep "Pages:" | awk '{print $2}')
    if [ -n "$PAGES" ]; then
        echo "Document pages: $PAGES"
    fi
fi

# Get file size
if [ -f "$MAIN_TEX.pdf" ]; then
    SIZE=$(du -h "$MAIN_TEX.pdf" | cut -f1)
    echo "PDF size: $SIZE"
fi

echo ""
echo "To view the PDF:"
echo "  - Linux: xdg-open $MAIN_TEX.pdf"
echo "  - macOS: open $MAIN_TEX.pdf"
echo "  - Windows: start $MAIN_TEX.pdf"
echo ""

# Clean up auxiliary files (optional)
read -p "Clean auxiliary files (.aux, .log, .bbl, etc.)? [y/N] " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    echo "Cleaning auxiliary files..."
    rm -f *.aux *.log *.bbl *.blg *.out *.toc *.lof *.lot *.synctex.gz
    echo "  ✓ Cleanup complete"
fi

echo ""
echo "================================================================================"

