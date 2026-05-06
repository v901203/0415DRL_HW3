import torch
import torch.nn as nn
import numpy as np
import random
import sys
import os
import pytorch_lightning as pl
from collections import deque
from torch.utils.data import DataLoader, Dataset

sys.path.append(os.path.join(os.path.dirname(__file__), 'repo', 'Chapter 3'))
from Gridworld import Gridworld

action_set = {0: 'u', 1: 'd', 2: 'l', 3: 'r'}

class RLDataset(Dataset):
    def __init__(self, replay_buffer, length=1000):
        self.replay_buffer = replay_buffer
        self.length = length

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        # Randomly sample from replay buffer
        return random.choice(self.replay_buffer)

class LitDQN(pl.LightningModule):
    def __init__(self, mem_size=1000, batch_size=200, gamma=0.9):
        super().__init__()
        self.save_hyperparameters()
        
        self.fc1 = nn.Linear(64, 150)
        self.fc2 = nn.Linear(150, 100)
        self.fc3 = nn.Linear(100, 4)
        
        # Training tip 1: Use Huber Loss (Smooth L1) instead of MSE to prevent large gradients
        self.loss_fn = nn.SmoothL1Loss() 
        self.replay = deque(maxlen=mem_size)
        
        self.epsilon = 1.0
        self.game = Gridworld(size=4, mode='random')
        self.state1 = self._get_state()
        
    def _get_state(self):
        state_ = self.game.board.render_np().reshape(1, 64) + np.random.rand(1, 64)/100.0
        return torch.from_numpy(state_).float()

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

    def play_step(self):
        qval = self(self.state1)
        qval_ = qval.detach().numpy()
        
        if random.random() < self.epsilon:
            action_ = np.random.randint(0, 4)
        else:
            action_ = np.argmax(qval_)
            
        action = action_set[action_]
        self.game.makeMove(action)
        
        state2 = self._get_state()
        reward = self.game.reward()
        done = True if reward != -1 else False
        
        self.replay.append((self.state1, action_, reward, state2, done))
        self.state1 = state2
        
        if done:
            self.game = Gridworld(size=4, mode='random')
            self.state1 = self._get_state()
            
        if self.epsilon > 0.1:
            self.epsilon -= 1/10000.0 # Decay epsilon over steps
            
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        # Training tip 2: Learning Rate Scheduling
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.9)
        return [optimizer], [scheduler]

    def train_dataloader(self):
        # Fill buffer before starting
        while len(self.replay) < self.hparams.batch_size:
            self.play_step()
        
        # Virtual epoch length of 100 iterations
        dataset = RLDataset(self.replay, length=100 * self.hparams.batch_size)
        return DataLoader(dataset, batch_size=self.hparams.batch_size)

    def training_step(self, batch, batch_idx):
        self.play_step() # Gather new experience
        
        s1, a, r, s2, d = batch
        s1 = s1.squeeze(1)
        s2 = s2.squeeze(1)
        r = r.to(torch.float32)
        d = d.to(torch.bool)
        
        Q1 = self(s1)
        with torch.no_grad():
            Q2 = self(s2)
            
        Y = r + self.hparams.gamma * torch.max(Q2, dim=1)[0] * (~d)
        X = Q1.gather(1, a.unsqueeze(1)).squeeze()
        
        loss = self.loss_fn(X, Y)
        self.log('train_loss', loss, prog_bar=True)
        return loss

if __name__ == '__main__':
    print("Training PyTorch Lightning DQN...")
    model = LitDQN()
    # Training tip 3: Gradient Clipping
    trainer = pl.Trainer(max_epochs=10, gradient_clip_val=1.0)
    trainer.fit(model)
