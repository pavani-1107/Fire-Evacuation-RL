# Fire Evacuation RL Agent

A reinforcement learning project that trains agents to evacuate from a building with dynamic fire spread using Q-Learning and SARSA algorithms.

## Project Overview

This project implements and compares two RL algorithms:
- **Q-Learning** (off-policy)
- **SARSA** (on-policy)

The agent learns to navigate a grid-based building environment to reach the exit while avoiding fire that spreads dynamically.

## Features

- Custom grid environment with fire spreading mechanics
- Q-Learning and SARSA implementations
- Algorithm comparison and analysis
- Hyperparameter tuning for both algorithms
- Training metrics and success rates

## Project Structure
```
fire-evacuation-rl/
├── env/
│   └── building_env.py              # Grid environment with fire dynamics
├── agents/
│   ├── qlearning.py                 # Q-Learning agent
│   └── sarsa.py                     # SARSA agent
├── training/
│   ├── train_qlearning.py           # Train Q-Learning
│   ├── train_sarsa.py               # Train SARSA
│   ├── compare_algorithms.py        # Compare both algorithms
│   ├── hyperparameter_tuning.py     # Tune Q-Learning hyperparameters
│   └── hyperparameter_tuning_sarsa.py # Tune SARSA hyperparameters
├── requirements.txt
└── README.md
```

## How to Run

### Setup
```bash
# Clone the repository
git clone https://github.com/pavani-1107/Fire-Evacuation-RL
cd fire-evacuation-rl

# Create virtual environment
python -m venv venv
source venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Train Agents
```bash
cd training

# Train Q-Learning
python train_qlearning.py

# Train SARSA
python train_sarsa.py

# Compare both algorithms
python compare_algorithms.py

# Hyperparameter tuning
python hyperparameter_tuning.py         # Q-Learning
python hyperparameter_tuning_sarsa.py   # SARSA
```

## Results

- **Q-Learning**: Achieves ~70-90% success rate, finds faster but riskier paths
- **SARSA**: Achieves ~65-85% success rate, learns safer and more conservative policies
- SARSA is more cautious around fire due to on-policy learning
- Q-Learning converges faster but takes more risks

## Hyperparameters

Tested combinations for both algorithms:
- **Learning rates (α)**: 0.1, 0.3, 0.5
- **Discount factors (γ)**: 0.9, 0.95, 0.99

Best results typically with α=0.1-0.3 and γ=0.99

## Environment Details

- **Grid Size**: 5×5 (configurable)
- **Agent Start**: Bottom-left corner
- **Exit**: Top-right corner
- **Fire Spread**: Probabilistic (10-30% per step)
- **Actions**: Up, Down, Left, Right, Stay
- **Rewards**: +100 (exit), -100 (fire), -1 (each step)