# Protein FT-Transformer for Multiclass Classification

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A state-of-the-art transformer-based model for multiclass protein classification, specifically designed for tabular protein datasets.

## Features

- **Advanced Architecture**: FT-Transformer optimized for tabular data
- **Multiclass Support**: Handles multiple protein classes efficiently
- **Feature Importance**: Provides interpretable feature importance scores
- **Class Imbalance Handling**: Focal loss + temperature scaling
- **Comprehensive Evaluation**: Multiple metrics and visualizations
- **Production Ready**: Easy deployment and inference

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/protein-ft-transformer.git
cd protein-ft-transformer

# Create virtual environment (optional but recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Install in development mode
pip install -e .
