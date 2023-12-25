!pip install gymnasium
# Import necessary libraries
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import gymnasium as gym
from collections import deque
import random

# Define the Q-network class using PyTorch
class QNetwork(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(QNetwork, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        return self.fc2(x)

# Define the Deep Q-Network (DQN) agent class
class DQNAgent:
    def __init__(self, state_size, action_size, hidden_size=258, gamma=0.99, epsilon_start=1.0, epsilon_end=0.01, epsilon_decay=0.995):
        # Initialize agent parameters
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=10000)
        self.gamma = gamma
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay

        # Initialize Q-network model and target model using the QNetwork class
        self.model = QNetwork(state_size, hidden_size, action_size)
        self.target_model = QNetwork(state_size, hidden_size, action_size)
        self.target_model.load_state_dict(self.model.state_dict())

        # Initialize Adam optimizer and mean squared error loss function
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)
        self.loss_function = nn.MSELoss()

    def select_action(self, state):
        # Epsilon-greedy action selection
        if np.random.rand() < self.epsilon:
            return np.random.choice(self.action_size)
        state = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
        q_values = self.model(state)
        return torch.argmax(q_values).item()

    def remember(self, state, action, reward, next_state, done):
        # Store experience tuple (state, action, reward, next_state, done) in replay memory
        self.memory.append((state, action, reward, next_state, done))

    def replay(self, batch_size):
        # Train on the whole memory ???
        minibatch = np.array(self.memory)
        states = np.vstack(minibatch[:, 0])
        actions = np.array(minibatch[:, 1], dtype=np.int64)
        rewards = np.array(minibatch[:, 2], dtype=np.int64)
        next_states = np.vstack(minibatch[:, 3])
        dones = np.array(minibatch[:, 4], dtype=np.int64)

        # Convert NumPy arrays to PyTorch tensors
        states = torch.tensor(states, dtype=torch.float32)
        next_states = torch.tensor(next_states, dtype=torch.float32)
        actions = torch.tensor(actions, dtype=torch.int64)
        rewards = torch.tensor(rewards, dtype=torch.float32)
        dones = torch.tensor(dones, dtype=torch.float32)

        # Compute Q-values for the current state and selected actions
        Q_values = self.model(states).gather(1, actions.unsqueeze(1))

        # Compute target Q-values for the next state using the target network
        next_Q_values = self.target_model(next_states).max(dim=1).values.detach()
        target_Q_values = rewards + (1 - dones) * self.gamma * next_Q_values

        # Update the Q-network using the temporal difference error
        self.optimizer.zero_grad()
        loss = self.loss_function(Q_values, target_Q_values.unsqueeze(1))
        loss.backward()
        self.optimizer.step()

        # Update epsilon for epsilon-greedy exploration
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)

    def update_target_model(self):
        # Update the target model by copying the parameters from the current model
        self.target_model.load_state_dict(self.model.state_dict())

# Define the training function for the DQN agent
def train_dqn(agent, env, episodes=1000, batch_size=1000):
    for episode in range(episodes):
        # Reset the environment and initialize variables for the current episode
        state = env.reset()
        total_reward = 0
        done = False

        # Run the episode until termination
        while not done:
            # Select an action, take a step, and store the experience in replay memory
            action = agent.select_action(state)
            next_state, reward, done, _ = env.step(action)
            agent.remember(state, action, reward, next_state, done)
            state = next_state
            total_reward += reward

        # Perform a Q-learning update using replay memory
        agent.replay(batch_size)
        # Update the target model to track the changes in the Q-network
        agent.update_target_model()

        # Print episode information every 10 episodes
        if episode % 10 == 0:
            print(f"Episode: {episode}, Total Reward: {total_reward}")

# Main execution block
if __name__ == "__main__":
    # Create the CartPole environment
    env = gym.make("CartPole-v1", render_mode='human')
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n

    # Initialize the DQN agent
    dqn_agent = DQNAgent(state_size, action_size)

    # Train the DQN agent on the CartPole environment
    train_dqn(dqn_agent, env)
