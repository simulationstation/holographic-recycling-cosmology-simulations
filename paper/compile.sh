#!/bin/bash
# Compile HRC paper
# Requires: pdflatex, bibtex

echo "Compiling HRC paper..."

# First pass
pdflatex -interaction=nonstopmode hrc_paper.tex

# Bibliography
bibtex hrc_paper

# Second pass (resolve references)
pdflatex -interaction=nonstopmode hrc_paper.tex

# Third pass (finalize)
pdflatex -interaction=nonstopmode hrc_paper.tex

echo "Done! Output: hrc_paper.pdf"
