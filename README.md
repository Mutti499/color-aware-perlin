
# Neural Network Initialization Using Color-Aware Perlin Noise

## Project Overview
This repository contains the implementation and improvement of neural network initialization using color-aware Perlin noise. The project consists of two main components:

### 1. Original Implementation (`perlin_original_code.py`)
- Custom implementation of the methodology described in the original paper
- Note: Results may differ from the original paper due to implementation details/code not shared in the publication
- Independent implementation focused on capturing the core concepts

### 2. Improved Version (`perlin_improved_code.py`)
- Enhanced implementation featuring the new `ColorPerlinNoiseDataset` class
- Significant improvements in performance and functionality
- Promising results that warrant further research and validation

## Configuration
The grid size can be customized by modifying the `N` and `M` parameters in the code:
```python
N = your_value  # Width of the grid
M = your_value  # Height of the grid
```

## Training
The optimizer parameters were determined using Optuna for optimal performance. These values are pre-configured in the code for the provided dataset.

## Future Development
The improved version shows potential for publication pending further validation and enhancement of the findings.
