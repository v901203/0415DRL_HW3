# Homework 3: DQN & Variants

This repository contains the implementation of Deep Q-Networks (DQN) and several of its advanced variants, applied to a Gridworld environment.

## 🚀 Overview & Results

### ❓ Q1: HW3-1 Naive DQN for static mode
**Task:** Run the provided code naive or Experience buffer reply. Submit a short understanding report.

**💡 Understanding Report:**
In `static` mode, all objects are fixed. The state is represented as a flattened 64-dimensional array.
- **Naive DQN:** Trains strictly online. Since consecutive frames are highly correlated, the learning process oscillates heavily and struggles to stabilize.
- **Experience Replay:** Solves instability by storing past experiences in a buffer and sampling mini-batches randomly. This breaks temporal correlation, drastically improving the smoothness and convergence of the loss curve.

**📊 Results:**
| Naive DQN Loss | Experience Replay DQN Loss |
|:---:|:---:|
| ![Naive DQN Loss](static/images/naive_dqn_loss.png) | ![ER DQN Loss](static/images/er_dqn_loss.png) |

---

### ❓ Q2: HW3-2 Enhanced DQN Variants for player mode
**Task:** Implement and compare Double DQN and Dueling DQN. Focus on how they improve upon the basic DQN approach.

**💡 Improvements Explanation:**
- **Double DQN:** Basic DQN suffers from overestimation bias because it uses the same network to select and evaluate actions. Double DQN decouples this: it uses the Main Network to select the action and the Target Network to evaluate it, yielding more accurate Q-values.
- **Dueling DQN:** Splits the network into a Value stream (how good the state is generally) and an Advantage stream (how much better an action is compared to others). This allows the network to learn which states are valuable without having to learn the effect of each action for every single state.

**📊 Results:**
| Double DQN Loss | Dueling DQN Loss |
|:---:|:---:|
| ![Double DQN Loss](static/images/double_dqn_loss.png) | ![Dueling DQN Loss](static/images/dueling_dqn_loss.png) |

---

### ❓ Q3: HW3-3 Enhance DQN for random mode WITH Training Tips
**Task:** Convert the DQN model from PyTorch to either Keras or PyTorch Lightning. Bonus points for integrating training techniques to stabilize/improve learning.

**💡 Implementation & Tips:**
We converted the model to **PyTorch Lightning** (`LitDQN` class) for the hardest `random` environment. To stabilize training in this chaotic mode, we integrated:
- ✨ **Huber Loss:** Replaced MSE with `nn.SmoothL1Loss()` to prevent exploding gradients when rewards fluctuate wildly.
- 📉 **Learning Rate Scheduling:** Added `StepLR` to gradually decay the learning rate, helping the model fine-tune its policy as it converges.
- ✂️ **Gradient Clipping:** Set `gradient_clip_val=1.0` in the Trainer to ensure updates remain within a stable bound.

**📊 Result Summary:**
The PyTorch Lightning model successfully encapsulated the training loop. By applying these tips, the model avoided catastrophic forgetting in the `random` environment where all objects (Player, Goal, Pit, Wall) spawn arbitrarily, successfully converging across epochs.

---

## 🛠 Setup & Requirements
To run the scripts locally, make sure you have the following Python packages installed:
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
