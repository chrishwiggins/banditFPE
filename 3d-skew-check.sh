#!/bin/bash
#
# 3D Probit Bandit: Gaussianity Analysis
#
# This script generates all figures for the 3D probit bandit analysis,
# studying convergence to Gaussianity of the distribution p(Y, A, G, t).
#
# Usage: ./3d-skew-check.sh
#
# Output figures (in fig/):
#   - probit_3d_gaussianity.png   : Marginal skewness & kurtosis (2x3 grid)
#   - probit_3d_coskewness.png    : Co-skewness and all 3rd cumulants
#   - probit_3d_moments.png       : Means and variances over time
#   - probit_3d_correlations.png  : Pairwise correlations Y-A, Y-G, A-G
#   - probit_3d_summary.png       : Comprehensive 2x4 summary figure
#

set -e  # Exit on error

echo "========================================================================"
echo "3D PROBIT BANDIT: GAUSSIANITY ANALYSIS"
echo "========================================================================"
echo ""
echo "This script runs the 3D probit bandit analysis to study convergence"
echo "to Gaussianity of the joint distribution p(Y, A, G, t)."
echo ""
echo "Sufficient statistics:"
echo "  Y = sum_t y_t       (total reward)"
echo "  A = sum_t a_t       (action imbalance)"
echo "  G = sum_t a_t*y_t   (reward-action correlation)"
echo ""
echo "Decision: P(a=+1) = Phi(beta * (GT - AY) / (T^2 - A^2))"
echo ""

# Navigate to src directory
cd "$(dirname "$0")/src"

echo "Running analysis..."
echo ""

python3 probit_bandit_3d.py

echo ""
echo "========================================================================"
echo "ANALYSIS COMPLETE"
echo "========================================================================"
echo ""
echo "Key findings:"
echo "  - Marginal distributions (Y, A, G) converge to Gaussian"
echo "  - Marginal skewnesses and excess kurtoses decay toward zero"
echo "  - Co-skewness decays more slowly than marginal skewnesses"
echo "  - CLT intuition applies: increments become approximately i.i.d."
echo ""
echo "See tex/probit_3d_gaussianity.tex for detailed writeup."
