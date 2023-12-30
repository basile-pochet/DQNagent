# Import necessary libraries
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import gym
from collections import deque
import random
import matplotlib.pyplot as plt


class QNetwork(nn.Module):
    """Q-network class using PyTorch.

    Attributes:
        fc1 (nn.Linear): First fully connected layer.
        fc2 (nn.Linear): Second fully connected layer.
        fc3 (nn.Linear): Third fully connected layer.
    """

    def __init__(self, input_size, hidden_size, output_size):
        """Initialize the QNetwork.

        Args:
            input_size (int): Size of the input.
            hidden_size (int): Size of the hidden layer.
            output_size (int): Size of the output.
        """
        super(QNetwork, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        """Forward pass of the QNetwork.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor.
        """
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)


class DQNAgent:
    """Deep Q-Network (DQN) agent class.

    Attributes:
        state_size (int): Size of the state space.
        action_size (int): Size of the action space.
        memory (deque): Replay memory.
        gamma (float): Discount factor used in the Q-values calculations.
        epsilon (float): Exploration-exploitation parameter.
        epsilon_end (float): Minimum value of epsilon.
        epsilon_decay (float): Decay factor for epsilon.
        model (QNetwork): Q-network model.
        target_model (QNetwork): Target Q-network model.
        optimizer (torch.optim.Adam): Adam optimizer.
        loss_function (nn.MSELoss): Mean Squared Error loss function.
    """

    def __init__(self, state_size, action_size, hidden_size=128, gamma=0.99, epsilon_start=1.0, epsilon_end=0.01, epsilon_decay=0.995):
        """Initialize the DQNAgent.

        Args:
            state_size (int): Size of the state space.
            action_size (int): Size of the action space.
            hidden_size (int): Size of the hidden layer.
            gamma (float): Discount factor.
            epsilon_start (float): Initial exploration-exploitation parameter.
            epsilon_end (float): Minimum value of epsilon.
            epsilon_decay (float): Decay factor for epsilon.
        """
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=10000)  # popleft() if maxlen is reached
        self.gamma = gamma
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay

        self.model = QNetwork(state_size, hidden_size, action_size)
        self.target_model = QNetwork(state_size, hidden_size, action_size)
        self.target_model.load_state_dict(self.model.state_dict())

        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)
        self.loss_function = nn.MSELoss()

    def select_action(self, state):
        """Select an action using epsilon-greedy strategy.
        It means that the more the agent plays, the less he is inclined to choose a random action.
        The is progressively giving more weight to the exploitation part and less to the exploration part.

        Args:
            state (np.ndarray): Current state.

        Returns:
            int: Selected action.
        """
        if np.random.rand() < self.epsilon:
            return np.random.choice(self.action_size)
        state = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
        q_values = self.model(state)
        return torch.argmax(q_values).item()

    def remember(self, state, action, reward, next_state, done):
        """Store experience tuple in replay memory.

        Args:
            state (np.ndarray): Current state.
            action (int): Taken action.
            reward (float): Received reward.
            next_state (np.ndarray): Next state reached with this action.
            done (bool): Termination flag (game is over/maximum attempts reached)
        """
        self.memory.append((state, action, reward, next_state, done))

    def replay(self, batch_size):
        """Perform a replay and update the Q-network.
        Calculations of the Q-values and the target Q-values with a random sample taken from the memory.
        Finding the best model according to the past actions.

        Args:
            batch_size (int): Size of the replay batch.
        """
        if len(self.memory) < batch_size:
            return
        minibatch = np.array(random.sample(self.memory, batch_size))

        states = np.vstack(minibatch[:, 0])
        actions = np.array(minibatch[:, 1], dtype=np.int64)
        rewards = np.array(minibatch[:, 2], dtype=np.int64)
        next_states = np.vstack(minibatch[:, 3])
        dones = np.array(minibatch[:, 4], dtype=np.int64)

        states = torch.tensor(states, dtype=torch.float32)
        next_states = torch.tensor(next_states, dtype=torch.float32)
        actions = torch.tensor(actions, dtype=torch.int64)
        rewards = torch.tensor(rewards, dtype=torch.float32)
        dones = torch.tensor(dones, dtype=torch.float32)

        Q_values = self.model(states).gather(1, actions.unsqueeze(1))

        next_Q_values = self.target_model(next_states).max(dim=1).values.detach()
        target_Q_values = rewards + (1 - dones) * self.gamma * next_Q_values

        self.optimizer.zero_grad()
        loss = self.loss_function(Q_values, target_Q_values.unsqueeze(1))
        loss.backward()
        self.optimizer.step()

        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay) #updating epsilon

    def update_target_model(self):
        """Update the target Q-network."""
        self.target_model.load_state_dict(self.model.state_dict())


def train_dqn(agent, env, episodes=1000, batch_size=10000, env_name="Unknown"):
    """Train the DQN agent.
    MAIN LOOP:
    The first step is to select an action (randomly or using the model, depending on the epsilon)
    Then we look at what this action implies: the next state, the reward and if the game is over or not.
    We update the state and the reward.
    This loop is repeated until the game is over (again, game can be over or we reached the maximum number of attempts).

    We repeat this operation for the number of episodes choosen, with an updated Network each time so the Agent has increasing rewards.

    At the end, we plot the results.

    Args:
        agent (DQNAgent): DQN agent.
        env (gym.Env): Gym environment.
        episodes (int): Number of episodes to train.
        batch_size (int): Size of the replay batch.
        env_name (str): Name of the environment. For the title of the plot.
    """
    all_rewards = []
    for episode in range(episodes): #loop for the number of episodes choosen
        state = env.reset()
        total_reward = 0
        done = False

        # Main loop
        while not done:
            action = agent.select_action(state)
            next_state, reward, done, _ = env.step(action)
            agent.remember(state, action, reward, next_state, done)
            state = next_state
            total_reward += reward

        agent.replay(batch_size)
        agent.update_target_model()
        all_rewards.append(total_reward)

    max_reward = max(all_rewards)
    mean_rewards = [sum(all_rewards[:i + 1]) / (i + 1) for i in range(len(all_rewards))]

    plt.plot(all_rewards, label='Total Rewards')
    plt.plot(mean_rewards, label='Mean Total Reward')
    plt.xlabel('Episode (times 10)')
    plt.ylabel('Total Reward')
    plt.title(f'Total and Mean Rewards over Episodes\n{env_name}\nSize of hidden layer: {agent.model.fc1.out_features}, Batch Size: {batch_size}')
    plt.text(len(all_rewards), max_reward, f'Max Reward: {max_reward}', verticalalignment='bottom')
    plt.legend()
    plt.show()


if __name__ == "__main__":
    env = gym.make('CartPole-v1')
    state_size = env.observation_space.shape[0] 
    action_size = env.action_space.n #change this to env.observation_space.shape[0] depending on the game, I could not find an other way to do it

    dqn_agent = DQNAgent(state_size, action_size, hidden_size=256 * 2)
    train_dqn(dqn_agent, env, env_name='CartPole-v1')
