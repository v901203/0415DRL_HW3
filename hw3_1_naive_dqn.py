import torch
import torch.nn as nn
import numpy as np
import random
import matplotlib.pyplot as plt
from collections import deque
import sys
import os

# Ensure the repo is in the path so we can import GridBoard correctly
sys.path.append(os.path.join(os.path.dirname(__file__), 'repo', 'Chapter 3'))
from Gridworld import Gridworld

class DQN(nn.Module):
    def __init__(self):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(64, 150)
        self.fc2 = nn.Linear(150, 100)
        self.fc3 = nn.Linear(100, 4)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

action_set = {0: 'u', 1: 'd', 2: 'l', 3: 'r'}

def train_naive_dqn(epochs=1000):
    model = DQN()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.MSELoss()
    gamma = 0.9
    epsilon = 1.0
    losses = []

    for i in range(epochs):
        game = Gridworld(size=4, mode='static')
        # Add slight noise to states to prevent identical exact values causing issues in simple NN
        state1_ = game.board.render_np().reshape(1, 64) + np.random.rand(1, 64)/100.0
        state1 = torch.from_numpy(state1_).float()
        status = 1
        
        while status == 1:
            qval = model(state1)
            qval_ = qval.data.numpy()
            
            if random.random() < epsilon:
                action_ = np.random.randint(0, 4)
            else:
                action_ = np.argmax(qval_)
                
            action = action_set[action_]
            game.makeMove(action)
            
            state2_ = game.board.render_np().reshape(1, 64) + np.random.rand(1, 64)/100.0
            state2 = torch.from_numpy(state2_).float()
            reward = game.reward()
            
            with torch.no_grad():
                newQ = model(state2)
            maxQ = torch.max(newQ)
            
            if reward == -1:
                Y = reward + (gamma * maxQ)
            else:
                Y = torch.tensor(reward, dtype=torch.float32)
                
            Y = torch.tensor(Y, dtype=torch.float32).unsqueeze(0)
            X = qval.squeeze()[action_].unsqueeze(0)
            
            loss = loss_fn(X, Y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            state1 = state2
            if reward != -1:
                status = 0
                
        if epsilon > 0.1:
            epsilon -= 1/epochs
            
        losses.append(loss.item())
        if i % 100 == 0:
            print(f"Naive DQN - Epoch {i}, Loss: {loss.item():.4f}, Epsilon: {epsilon:.4f}")

    return model, losses

def train_experience_replay(epochs=1000):
    model = DQN()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.MSELoss()
    gamma = 0.9
    epsilon = 1.0
    losses = []
    
    # Experience Replay Buffer
    mem_size = 1000
    batch_size = 200
    replay = deque(maxlen=mem_size)
    
    for i in range(epochs):
        game = Gridworld(size=4, mode='static')
        state1_ = game.board.render_np().reshape(1, 64) + np.random.rand(1, 64)/100.0
        state1 = torch.from_numpy(state1_).float()
        status = 1
        
        while status == 1:
            qval = model(state1)
            qval_ = qval.data.numpy()
            
            if random.random() < epsilon:
                action_ = np.random.randint(0, 4)
            else:
                action_ = np.argmax(qval_)
                
            action = action_set[action_]
            game.makeMove(action)
            
            state2_ = game.board.render_np().reshape(1, 64) + np.random.rand(1, 64)/100.0
            state2 = torch.from_numpy(state2_).float()
            reward = game.reward()
            
            done = True if reward != -1 else False
            replay.append((state1, action_, reward, state2, done))
            
            if len(replay) > batch_size:
                minibatch = random.sample(replay, batch_size)
                state1_batch = torch.cat([s1 for (s1, a, r, s2, d) in minibatch])
                action_batch = torch.tensor([a for (s1, a, r, s2, d) in minibatch])
                reward_batch = torch.tensor([r for (s1, a, r, s2, d) in minibatch], dtype=torch.float32)
                state2_batch = torch.cat([s2 for (s1, a, r, s2, d) in minibatch])
                done_batch = torch.tensor([d for (s1, a, r, s2, d) in minibatch], dtype=torch.bool)
                
                Q1 = model(state1_batch)
                with torch.no_grad():
                    Q2 = model(state2_batch)
                
                Y = reward_batch + gamma * torch.max(Q2, dim=1)[0] * (~done_batch)
                X = Q1.gather(1, action_batch.unsqueeze(1)).squeeze()
                
                loss = loss_fn(X, Y)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                losses.append(loss.item())
            
            state1 = state2
            if done:
                status = 0
                
        if epsilon > 0.1:
            epsilon -= 1/epochs
            
        if i % 100 == 0:
            avg_loss = np.mean(losses[-100:]) if len(losses) > 0 else 0
            print(f"ER DQN - Epoch {i}, Avg Loss: {avg_loss:.4f}, Epsilon: {epsilon:.4f}")

    return model, losses

if __name__ == '__main__':
    print("Training Naive DQN...")
    train_naive_dqn(epochs=1000)
    print("\nTraining DQN with Experience Replay...")
    train_experience_replay(epochs=1000)
