import torch
import torch.nn as nn
import numpy as np
import random
from collections import deque
import sys
import os

# Ensure the repo is in the path so we can import GridBoard correctly
sys.path.append(os.path.join(os.path.dirname(__file__), 'repo', 'Chapter 3'))
from Gridworld import Gridworld

action_set = {0: 'u', 1: 'd', 2: 'l', 3: 'r'}

class DoubleDQN(nn.Module):
    def __init__(self):
        super(DoubleDQN, self).__init__()
        self.fc1 = nn.Linear(64, 150)
        self.fc2 = nn.Linear(150, 100)
        self.fc3 = nn.Linear(100, 4)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

class DuelingDQN(nn.Module):
    def __init__(self):
        super(DuelingDQN, self).__init__()
        self.fc1 = nn.Linear(64, 150)
        self.fc2 = nn.Linear(150, 100)
        
        # Value stream
        self.value_stream = nn.Linear(100, 1)
        # Advantage stream
        self.advantage_stream = nn.Linear(100, 4)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        
        val = self.value_stream(x)
        adv = self.advantage_stream(x)
        
        # Q(s, a) = V(s) + (A(s, a) - mean(A(s, a)))
        q_vals = val + (adv - adv.mean(dim=1, keepdim=True))
        return q_vals

def train_double_dqn(epochs=1000):
    model = DoubleDQN()
    target_model = DoubleDQN()
    target_model.load_state_dict(model.state_dict())
    
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.MSELoss()
    gamma = 0.9
    epsilon = 1.0
    losses = []
    
    mem_size = 1000
    batch_size = 200
    replay = deque(maxlen=mem_size)
    sync_freq = 500
    j = 0
    
    for i in range(epochs):
        game = Gridworld(size=4, mode='player')
        state1_ = game.board.render_np().reshape(1, 64) + np.random.rand(1, 64)/100.0
        state1 = torch.from_numpy(state1_).float()
        status = 1
        
        while status == 1:
            j += 1
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
                
                # Double DQN logic
                with torch.no_grad():
                    Q2_main = model(state2_batch)
                    best_actions = torch.argmax(Q2_main, dim=1)
                    Q2_target = target_model(state2_batch)
                    
                    target_q = Q2_target.gather(1, best_actions.unsqueeze(1)).squeeze()
                
                Y = reward_batch + gamma * target_q * (~done_batch)
                X = Q1.gather(1, action_batch.unsqueeze(1)).squeeze()
                
                loss = loss_fn(X, Y)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                losses.append(loss.item())
                
                if j % sync_freq == 0:
                    target_model.load_state_dict(model.state_dict())
            
            state1 = state2
            if done:
                status = 0
                
        if epsilon > 0.1:
            epsilon -= 1/epochs
            
        if i % 100 == 0:
            avg_loss = np.mean(losses[-100:]) if len(losses) > 0 else 0
            print(f"Double DQN - Epoch {i}, Avg Loss: {avg_loss:.4f}, Epsilon: {epsilon:.4f}")

    return model, losses


def train_dueling_dqn(epochs=1000):
    model = DuelingDQN()
    target_model = DuelingDQN()
    target_model.load_state_dict(model.state_dict())
    
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.MSELoss()
    gamma = 0.9
    epsilon = 1.0
    losses = []
    
    mem_size = 1000
    batch_size = 200
    replay = deque(maxlen=mem_size)
    sync_freq = 500
    j = 0
    
    for i in range(epochs):
        game = Gridworld(size=4, mode='player')
        state1_ = game.board.render_np().reshape(1, 64) + np.random.rand(1, 64)/100.0
        state1 = torch.from_numpy(state1_).float()
        status = 1
        
        while status == 1:
            j += 1
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
                    Q2 = target_model(state2_batch)
                
                Y = reward_batch + gamma * torch.max(Q2, dim=1)[0] * (~done_batch)
                X = Q1.gather(1, action_batch.unsqueeze(1)).squeeze()
                
                loss = loss_fn(X, Y)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                losses.append(loss.item())
                
                if j % sync_freq == 0:
                    target_model.load_state_dict(model.state_dict())
            
            state1 = state2
            if done:
                status = 0
                
        if epsilon > 0.1:
            epsilon -= 1/epochs
            
        if i % 100 == 0:
            avg_loss = np.mean(losses[-100:]) if len(losses) > 0 else 0
            print(f"Dueling DQN - Epoch {i}, Avg Loss: {avg_loss:.4f}, Epsilon: {epsilon:.4f}")

    return model, losses


if __name__ == '__main__':
    print("Training Double DQN...")
    train_double_dqn(epochs=1000)
    print("\nTraining Dueling DQN...")
    train_dueling_dqn(epochs=1000)
