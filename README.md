# Homework 3: DQN & Variants

This repository contains the implementation of Deep Q-Networks (DQN) and several of its advanced variants, applied to a Gridworld environment.

## 🚀 Overview
The project implements reinforcement learning agents across three environment difficulties:
1. **Static Mode (`hw3_1_naive_dqn.py`)**: 
   - Implements a Naive Online DQN.
   - Implements a DQN with **Experience Replay Buffer** to stabilize training.
2. **Player Mode (`hw3_2_advanced_dqn.py`)**:
   - Implements **Double DQN** to combat overestimation bias by decoupling action selection and evaluation.
   - Implements **Dueling DQN** to separate State-Value and Action-Advantage estimations.
3. **Random Mode (`hw3_3_lightning_dqn.py`)**:
   - Uses **PyTorch Lightning** for robust training in the hardest environment where all objects spawn randomly.
   - Incorporates advanced training tips: Huber Loss, Learning Rate Scheduling (StepLR), and Gradient Clipping.

## 📊 Results Showcase
We have provided an interactive HTML showcase of the training processes and loss curves.
You can view the full documentation and plotted results by opening:
👉 `index.html`

## 🛠 Setup & Requirements
To run the scripts, make sure you have the following Python packages installed:
```bash
pip install torch torchvision torchaudio pytorch-lightning matplotlib numpy
```

## 🏃 How to Run
```bash
# Run Naive DQN & Experience Replay
python hw3_1_naive_dqn.py

# Run Double DQN & Dueling DQN
python hw3_2_advanced_dqn.py

# Run PyTorch Lightning DQN
python hw3_3_lightning_dqn.py
```
