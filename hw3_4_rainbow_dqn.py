import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random
import math
import sys
import os
import matplotlib.pyplot as plt

# Ensure the repo is in the path so we can import Gridworld correctly
sys.path.append(os.path.join(os.path.dirname(__file__), 'repo', 'Chapter 3'))
from Gridworld import Gridworld

action_set = {0: 'u', 1: 'd', 2: 'l', 3: 'r'}

# ==========================================
# Component 1: Noisy Networks for Exploration
# ==========================================
class NoisyLinear(nn.Module):
    def __init__(self, in_features, out_features, std_init=0.5):
        super(NoisyLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.std_init = std_init
        
        self.weight_mu = nn.Parameter(torch.empty(out_features, in_features))
        self.weight_sigma = nn.Parameter(torch.empty(out_features, in_features))
        self.register_buffer('weight_epsilon', torch.empty(out_features, in_features))
        
        self.bias_mu = nn.Parameter(torch.empty(out_features))
        self.bias_sigma = nn.Parameter(torch.empty(out_features))
        self.register_buffer('bias_epsilon', torch.empty(out_features))
        
        self.reset_parameters()
        self.reset_noise()

    def reset_parameters(self):
        mu_range = 1 / math.sqrt(self.in_features)
        self.weight_mu.data.uniform_(-mu_range, mu_range)
        self.weight_sigma.data.fill_(self.std_init / math.sqrt(self.in_features))
        self.bias_mu.data.uniform_(-mu_range, mu_range)
        self.bias_sigma.data.fill_(self.std_init / math.sqrt(self.out_features))

    def _scale_noise(self, size):
        x = torch.randn(size)
        return x.sign().mul_(x.abs().sqrt_())

    def reset_noise(self):
        epsilon_in = self._scale_noise(self.in_features)
        epsilon_out = self._scale_noise(self.out_features)
        self.weight_epsilon.copy_(epsilon_out.ger(epsilon_in))
        self.bias_epsilon.copy_(epsilon_out)

    def forward(self, x):
        if self.training:
            weight = self.weight_mu + self.weight_sigma * self.weight_epsilon
            bias = self.bias_mu + self.bias_sigma * self.bias_epsilon
        else:
            weight = self.weight_mu
            bias = self.bias_mu
        return F.linear(x, weight, bias)

# ==========================================
# Component 2 & 3: Dueling & Categorical (C51)
# ==========================================
class RainbowNet(nn.Module):
    def __init__(self, num_actions=4, num_atoms=51, v_min=-10.0, v_max=10.0):
        super(RainbowNet, self).__init__()
        self.num_actions = num_actions
        self.num_atoms = num_atoms
        self.v_min = v_min
        self.v_max = v_max
        
        # Support of the distribution
        self.register_buffer('support', torch.linspace(v_min, v_max, num_atoms))
        
        self.fc1 = nn.Linear(64, 128)
        self.fc2 = nn.Linear(128, 128)
        
        # Dueling streams with Noisy layers
        self.value_hidden = NoisyLinear(128, 128)
        self.value_out = NoisyLinear(128, num_atoms)
        
        self.adv_hidden = NoisyLinear(128, 128)
        self.adv_out = NoisyLinear(128, num_actions * num_atoms)

    def forward(self, x):
        batch_size = x.size(0)
        
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        
        # Value stream
        val_hid = torch.relu(self.value_hidden(x))
        val = self.value_out(val_hid).view(batch_size, 1, self.num_atoms)
        
        # Advantage stream
        adv_hid = torch.relu(self.adv_hidden(x))
        adv = self.adv_out(adv_hid).view(batch_size, self.num_actions, self.num_atoms)
        
        # Combine using Dueling formula
        q_dist = val + adv - adv.mean(dim=1, keepdim=True)
        # Apply softmax to get probabilities (Categorical)
        prob = F.softmax(q_dist, dim=-1)
        return prob

    def get_action(self, x):
        prob = self.forward(x)
        expected_q = (prob * self.support).sum(dim=2)
        return expected_q.argmax(dim=1).item()
        
    def reset_noise(self):
        self.value_hidden.reset_noise()
        self.value_out.reset_noise()
        self.adv_hidden.reset_noise()
        self.adv_out.reset_noise()

# ==========================================
# Component 4: Prioritized Experience Replay (PER)
# ==========================================
class PrioritizedReplayBuffer:
    def __init__(self, capacity, alpha=0.6):
        self.capacity = capacity
        self.alpha = alpha
        self.buffer = []
        self.priorities = np.zeros((capacity,), dtype=np.float32)
        self.pos = 0

    def push(self, state, action, reward, next_state, done):
        max_prio = self.priorities.max() if self.buffer else 1.0
        
        if len(self.buffer) < self.capacity:
            self.buffer.append((state, action, reward, next_state, done))
        else:
            self.buffer[self.pos] = (state, action, reward, next_state, done)
            
        self.priorities[self.pos] = max_prio
        self.pos = (self.pos + 1) % self.capacity

    def sample(self, batch_size, beta=0.4):
        if len(self.buffer) == self.capacity:
            prios = self.priorities
        else:
            prios = self.priorities[:self.pos]
            
        probs = prios ** self.alpha
        probs /= probs.sum()
        
        indices = np.random.choice(len(self.buffer), batch_size, p=probs)
        samples = [self.buffer[idx] for idx in indices]
        
        total = len(self.buffer)
        weights = (total * probs[indices]) ** (-beta)
        weights /= weights.max()
        
        states, actions, rewards, next_states, dones = zip(*samples)
        return states, actions, rewards, next_states, dones, indices, weights

    def update_priorities(self, indices, priorities):
        for idx, prio in zip(indices, priorities):
            self.priorities[idx] = prio

# ==========================================
# Projection logic for C51
# ==========================================
def project_distribution(next_states, rewards, dones, model, target_model, gamma, support):
    batch_size = next_states.size(0)
    num_atoms = support.size(0)
    v_min, v_max = support[0].item(), support[-1].item()
    delta_z = float(v_max - v_min) / (num_atoms - 1)
    
    # Component 5: Double DQN for action selection
    with torch.no_grad():
        next_prob_main = model(next_states)
        next_q_main = (next_prob_main * support).sum(2)
        next_actions = next_q_main.argmax(1)
        
        next_prob_target = target_model(next_states)
        next_prob_target = next_prob_target[range(batch_size), next_actions, :]
        
    # Compute the projection
    tz = rewards.unsqueeze(1) + (1 - dones.unsqueeze(1)) * gamma * support.unsqueeze(0)
    tz = tz.clamp(min=v_min, max=v_max)
    
    b = (tz - v_min) / delta_z
    l = b.floor().long()
    u = b.ceil().long()
    
    # Handle exact matches (l == u)
    l[(u > 0) * (l == u)] -= 1
    u[(l < (num_atoms - 1)) * (l == u)] += 1
    
    m = torch.zeros(batch_size, num_atoms)
    offset = torch.linspace(0, ((batch_size - 1) * num_atoms), batch_size).long().unsqueeze(1).expand(batch_size, num_atoms)
    
    m.view(-1).index_add_(0, (l + offset).view(-1), (next_prob_target * (u.float() - b)).view(-1))
    m.view(-1).index_add_(0, (u + offset).view(-1), (next_prob_target * (b - l.float())).view(-1))
    
    return m

# ==========================================
# Main Training Loop
# ==========================================
def train_rainbow(epochs=1000):
    # Hyperparameters
    mem_size = 1000
    batch_size = 64 # Smaller batch size to help PER and C51
    gamma = 0.99
    # Component 6: Multi-step returns (N-step)
    n_step = 3
    sync_freq = 200
    
    model = RainbowNet()
    target_model = RainbowNet()
    target_model.load_state_dict(model.state_dict())
    
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    buffer = PrioritizedReplayBuffer(mem_size)
    
    losses = []
    rewards_log = []
    
    j = 0
    beta = 0.4
    
    for i in range(epochs):
        game = Gridworld(size=4, mode='random')
        state = torch.from_numpy(game.board.render_np().reshape(1, 64) + np.random.rand(1, 64)/100.0).float()
        
        status = 1
        n_step_buffer = []
        episode_reward = 0
        step_count = 0
        
        while status == 1 and step_count < 100:
            j += 1
            step_count += 1
            
            # Action selection is handled by NoisyNets
            model.reset_noise() # Reset noise to encourage diverse actions in a single episode
            action_idx = model.get_action(state)
            action = action_set[action_idx]
            
            game.makeMove(action)
            reward = game.reward()
            episode_reward += reward
            done = True if reward != -1 else False
            
            next_state = torch.from_numpy(game.board.render_np().reshape(1, 64) + np.random.rand(1, 64)/100.0).float()
            
            n_step_buffer.append((state, action_idx, reward, next_state, done))
            
            if len(n_step_buffer) == n_step or done:
                # Calculate N-step return
                n_reward = sum([transition[2] * (gamma ** idx) for idx, transition in enumerate(n_step_buffer)])
                n_state = n_step_buffer[0][0]
                n_action = n_step_buffer[0][1]
                n_next_state = n_step_buffer[-1][3]
                n_done = n_step_buffer[-1][4]
                
                buffer.push(n_state, n_action, n_reward, n_next_state, n_done)
                
                # Slide window
                if not done:
                    n_step_buffer.pop(0)
            
            state = next_state
            
            if len(buffer.buffer) >= batch_size:
                beta = min(1.0, beta + (1.0 - 0.4) / epochs) # anneal beta
                
                states, actions, rewards, next_states, dones, indices, weights = buffer.sample(batch_size, beta)
                
                states = torch.cat(states)
                actions = torch.tensor(actions)
                rewards = torch.tensor(rewards).float()
                next_states = torch.cat(next_states)
                dones = torch.tensor(dones).float()
                weights = torch.tensor(weights).float()
                
                # Reset noise for training
                model.reset_noise()
                target_model.reset_noise()
                
                # Get current distribution
                current_prob = model(states)[range(batch_size), actions, :]
                
                # Target distribution using Multi-step Gamma (gamma^n_step)
                m = project_distribution(next_states, rewards, dones, model, target_model, gamma ** n_step, model.support)
                
                # Cross-entropy loss
                loss = -(m * current_prob.add(1e-8).log()).sum(1)
                
                # Update PER priorities based on loss
                buffer.update_priorities(indices, loss.detach().numpy() + 1e-6)
                
                # Apply PER weights
                loss = (loss * weights).mean()
                
                optimizer.zero_grad()
                loss.backward()
                # Clip gradients for stability
                torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
                optimizer.step()
                
                losses.append(loss.item())
                
                if j % sync_freq == 0:
                    target_model.load_state_dict(model.state_dict())
            
            if done:
                status = 0
                rewards_log.append(episode_reward)
                
        if i % 100 == 0:
            avg_loss = np.mean(losses[-100:]) if len(losses) > 0 else 0
            avg_reward = np.mean(rewards_log[-100:]) if len(rewards_log) > 0 else 0
            print(f"Rainbow DQN - Epoch {i}, Avg Loss: {avg_loss:.4f}, Avg Reward: {avg_reward:.2f}")

    print("Training finished! Saving plots...")
    
    # Ensure static/images dir exists
    os.makedirs(os.path.join(os.path.dirname(__file__), 'static', 'images'), exist_ok=True)
    
    # Plot Loss
    plt.figure(figsize=(10, 5))
    plt.plot(losses, color='purple', alpha=0.7)
    plt.title('Rainbow DQN Loss')
    plt.xlabel('Training Steps')
    plt.ylabel('Loss')
    plt.savefig(os.path.join(os.path.dirname(__file__), 'static', 'images', 'rainbow_dqn_loss.png'))
    plt.close()

if __name__ == '__main__':
    print("Starting Full Rainbow DQN Training in Random Mode GridWorld...")
    train_rainbow(epochs=1000)
