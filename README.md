# CarRacing PPO Agent

A reinforcement learning project that trains an agent to solve the CarRacing-v3 environment from Gymnasium using Proximal Policy Optimization (PPO) with Stable Baselines3.

## Features

- Train a PPO agent on the CarRacing-v3 environment
- Save and load trained models
- Evaluate agent performance with rendering and reward statistics

## Installation

1. (Optional) Create and activate a virtual environment:
    ```bash
    python3 -m venv venv
    source venv/bin/activate
    ```
2. Install dependencies:
    ```bash
    pip install -r requirements.txt
    ```

## Usage

### Train the Agent

```bash
python train.py
```
- Trains a PPO agent for 1,000,000 timesteps.
- Saves the trained model to `models/ppo_carracing`.
- Training logs are saved in the `logs/` directory for TensorBoard visualization.

### Evaluate the Agent

```bash
python evaluate.py
```
- Loads the trained model from `models/ppo_carracing`.
- Runs 10 evaluation episodes with rendering enabled.
- Prints average, max, and min rewards.

## Directory Structure

```
.
├── train.py              # Training script
├── evaluate.py           # Evaluation script
├── requirements.txt      # Python dependencies
├── models/               # Saved models
├── logs/                 # Training logs (TensorBoard)
├── vec_normalize.pkl     # (If used) Normalization statistics
```

## Requirements

- Python 3.8+
- [gymnasium](https://github.com/Farama-Foundation/Gymnasium)
- [stable-baselines3](https://github.com/DLR-RM/stable-baselines3)
- torch (installed as a dependency of stable-baselines3)

Install all dependencies with:
```bash
pip install -r requirements.txt
```

## License

MIT
