# CFLight Project

## Overview
This repository contains the implementation of the CFLight project, a framework for traffic light optimization and related experiments. It includes various Python scripts for training, testing, and visualization.

## Prerequisites
- Python 3.10
- Required packages listed in `requirements.txt`

## Installation
1. Ensure Python 3.10 is installed on your system.
2. Install the required dependencies by running:
   ```bash
   pip install -r requirements.txt
   ```
   Note: This project also supports additional packages like SUMO, which should be installed separately if not included in `requirements.txt`.

## Usage
To run the main CFLight script, execute:
```bash
python CFLight.py
```

## File Structure
- `CFLight_ablation.py`: Ablation study script for CFLight.
- `CFLight_causal_vision.py`: Causal vision implementation for CFLight.
- `CFLight_loss.py`: Loss function definitions for CFLight.
- `CFLight_q.py`: Q-learning or related module.
- `CFLight_r.py`: Reward function or related module.
- `SCM_Experiments.py`: Experiments with Structural Causal Models.
- `buffer.py`: Buffer implementation (likely for reinforcement learning).
- `draw.py`: Visualization or drawing utilities.
- `gan_cf.py`: Generative Adversarial Network for counterfactuals.
- `gan_cf_loss.py`: Loss functions for GAN counterfactuals.
- `model.py`: Main model definition.
- `model_loss.py`: Loss functions for the model.
- `plt.py`: Plotting utilities.
- `plt_ab.py`: Ablation plotting script.
- `readme.md`: This file.
- `requirements.txt`: List of Python dependencies.
- `safe_act.py`: Safe action selection module.
- `safe_loss.py`: Safe loss function implementation.

## Notes
- Ensure SUMO is properly configured if used in your experiments.
- Refer to individual script comments for detailed usage instructions.
