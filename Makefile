# Makefile for Bandit FPE Analysis
# Generates figures and compiles LaTeX documents
#
# Usage:
#   make all      - Generate figures and compile all LaTeX documents
#   make figures  - Generate all Python figures
#   make tex      - Compile all LaTeX documents
#   make clean    - Remove generated files
#   make help     - Show this help message

PYTHON := python3
LATEX := pdflatex

# Directories
SRC_DIR := src
TEX_DIR := tex
FIG_DIR := fig

# Source files
FIGURE_SCRIPT := $(SRC_DIR)/generate_all_figures.py
TEX_SOURCES := $(wildcard $(TEX_DIR)/*.tex)
PDF_OUTPUTS := $(TEX_SOURCES:.tex=.pdf)

# Figure outputs
FIGURES := $(FIG_DIR)/late_time_gaussian_convergence.png \
           $(FIG_DIR)/truncation_rate_error.png \
           $(FIG_DIR)/rates_comparison.png

.PHONY: all figures tex clean help

all: figures tex

# Generate all figures from Python
figures: $(FIGURES)

$(FIGURES): $(FIGURE_SCRIPT) $(SRC_DIR)/truncation_rate_error.py $(SRC_DIR)/edgeworth_cumulants.py
	@echo "Generating figures..."
	cd $(SRC_DIR) && $(PYTHON) generate_all_figures.py

# Compile LaTeX documents
tex: $(PDF_OUTPUTS)

$(TEX_DIR)/%.pdf: $(TEX_DIR)/%.tex $(FIGURES)
	@echo "Compiling $<..."
	cd $(TEX_DIR) && $(LATEX) -interaction=nonstopmode $(notdir $<) || true
	cd $(TEX_DIR) && $(LATEX) -interaction=nonstopmode $(notdir $<) || true

# Clean generated files
clean:
	@echo "Cleaning generated files..."
	rm -f $(TEX_DIR)/*.aux $(TEX_DIR)/*.log $(TEX_DIR)/*.out
	rm -f $(TEX_DIR)/*.pdf
	rm -f $(FIG_DIR)/*.png

# Help target
help:
	@echo "Bandit FPE Analysis - Makefile targets:"
	@echo ""
	@echo "  make all      - Generate figures and compile LaTeX documents"
	@echo "  make figures  - Generate all Python figures"
	@echo "  make tex      - Compile all LaTeX documents (requires figures)"
	@echo "  make clean    - Remove generated files"
	@echo "  make help     - Show this help message"
	@echo ""
	@echo "Dependencies:"
	@echo "  - Python 3 with numpy, scipy, matplotlib"
	@echo "  - pdflatex (for tex target)"
	@echo ""
	@echo "Directory structure:"
	@echo "  src/  - Python source code"
	@echo "  tex/  - LaTeX documents"
	@echo "  fig/  - Generated figures"
