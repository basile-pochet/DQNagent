!pip install gymnasium
import gymnasium as gym
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

# Defining a basic linear network
class Linear_Network(nn.Module):
    def __init__(self, input_size, output_size):
        super(Linear_Network, self).__init__()
        self.linear1 = nn.Linear(input_size, hidden_size)
        self.linear2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = F.relu(self.linear1(x))
        x = self.linear2(x)
        return x

MAX_MEMORY = 100000

# Defining the DQN Agent
class Agent:
    def __init__(self, state_dim, action_dim, learning_rate, gamma, epsilon):
        self.state_dim = state_dim
        self.action_dim= action_dim
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.epsilon = epsilon
        self.memory = deque(maxlen=MAX_MEMORY) # popleft()
        self.model = Linear_Network(state_dim, 256, action_dim)

    def get_state(self,game):
      pass
      
