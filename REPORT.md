# Deep Q-Learning

## Introduction

If we want to talk about Deep Q-Learning we need to introduce Reinforcement Learning, Deep Q-Learning being a type of RL.
Reinforcement learning is a type of machine learning that trains algorithms to learn from their environments through trial and error. To do that, an agent is given information about its environment and do an action accordingly. It receives then a reward or a punishment so it can optimize its behaviors over time. The data is ccumulated through the agent's interactions with the environment. In other words, "reinforcement learning is teaching a software agent how to behave in an environment by telling it how good it's doing" ([Patrick Loeber](https://www.youtube.com/watch?v=L8ypSXwyBds)).
Reinforcement learning is being used in various domains such as game playing, autonomous systems, robotics etc and is a powerful approach for training intelligent agents.

## Deep Q-Learning

Deep Q-Learning is a type of reinforcement learning where we use a deep neural network to predict the actions. 

### Key Concepts

#### Q-Value

Q value is standing for quality of action, it represents the reward we obtain if we play this action while being at this state. 

The basic main loop we have to understand for DQL is the following (after initializing the Q-value): 

- choose the action (from the state)
- perform the action
- measure the reward
- update Q-value and train the model.

The last step, training the model, needs a loss function. It is the reason why we use the **Bellman Equation**:

$$Q_{new}(s_t, a_t) = (1 - \alpha) \cdot Q_{current}(s_t, a_t) + \alpha \cdot \left(r_{t+1} + \gamma \cdot \max_a Q(s_{t+1}, a)\right)$$

With 
- $\alpha$ the learning rate, 0 makes the agent exploit only prior knowleadge and 1 makes the agent consider only the most recent information
- *r_{t+1}* the reward received when moving from state *S_t* to *S_{t+1}*
- $\gamma$ the discount factor, 0 makes the agent focus only on current rewards (immediate rewards) and 1 makes it focus only on long term rewards. 
- $\max_a Q(s_{t+1}, a)$ the estimate of optimal future value

#### Q-Learning

Q-Learning is an off-policy algorithm that learns about the greedy policy $a = \max_{a} Q(s, a; \theta)$ while using a different behavior policy for acting in the environment/collecting data. The basic idea behind Q-Learning is to use the Bellman optimality equation as an iterative update $Q_{i}(s, a) \leftarrow \mathbb{E}\left[ r + \gamma \max_{a'} Q_{i}(s', a')\right]$, and it can be shown that this converges to the optimal (Q)-function, i.e. $Q_i \rightarrow Q^*$ as $i \rightarrow \infty$. For most problems, it is impractical to represent the (Q)-function as a table containing values for each combination. Instead, a function approximator, such as a neural network with parameters $\theta$, is trained to estimate the Q-values, i.e. $Q(s, a; \theta)$ 


#### Experience replay

Experience Replay is a technique used to make the network updates more stable. It involves storing the transitions that the agent observes, allowing the data to be reused later. By sampling from it randomly, the transitions that build up a batch are decorrelated, which greatly stabilizes and improves the DQN training procedure


#### Target Networks

To stabilize the training of the Q network, a separate neural network called the Target network is used. It is a copy of the Q network that is updated less frequently to provide more stable target values during the training process. The parameters from the previous iteration are fixed and not updated. In practice, a snapshot of the network parameters from a few iterations ago is used instead of the current parameters. This copy is called the target network.


#### Exploration vs. Exploitation (Epsilon-Greedy)

The exploration-exploitation tradeoff is a very important concept in reinforcement learning. A good strategy can improve learning speed and the final total reward. The agent uses an $\epsilon$-greedy policy that selects the greedy action with probability $1 - \epsilon$ and selects a random action with probability $\epsilon$, where $\epsilon$ is the exploration rate. This allows the agent to balance between exploring new actions and exploiting the current best-known actions

In our case, we use the $\epsilon$ as a decreasing value. The more the agent is going to play, the smaller the $\epsilon$ is going to be. The agent decides to play a random action if the command np.random.rand() is smaller than $\epsilon$.

### Architecture

The Neural Network used can be a simple Linear Neural Network, which takes as inputs the state and outputs the Q-Value for each action.

The Q network and the Target network have the same architecture but different weights. The Q network is updated frequently during training, while the Target network is updated less often to provide more stable target values.
The use of two neural networks, along with Experience Replay, is a key architectural choice in DQL that contributes to its stability and effectiveness in learning from the environment

### Training Process

Now the most important part, how do we train a DQN Agent ? There are a few steps.

  - Initialization

We first initialize the Q network and the target network (which is a copy of the first one) with random weights. In addition, we fix the size of the replay memory. 

  - Sampling

The agent performs an action and stores the observed result in the memory. Most of the time the result consists of: (state, action, reward, next state, done).

  - Training

This is the most important part, the agent select a random batch from the memory and uses it to update the Q-network using a gradient descent update step.

It then uses the Mean Squared Error to compare the Q-value predicion and target so it can update the Q-network's weights.  


# Munchausen agent

The Munchausen Reinforcement Learning (M-DQN) is a modification of the Deep Q-Learning (DQN) algorithm that incorporates a term inspired by the Munchausen game. This modification is a simple yet powerful extension of existing agents, and it is theoretically grounded, performing implicit KL regularization and enjoying a very strong performance bound. The M-DQN agent has been shown to outperform the traditional DQN agent in various settings, including Atari games, and it has been compared to other state-of-the-art agents such as C51 and Rainbow, demonstrating superior performance. The M-DQN agent achieves this by increasing the action gap and leveraging the Munchausen term, which leads to significant improvements in learning efficiency and overall performance. The M-DQN agent is theoretically sound and does not rely on common deep reinforcement learning tricks, making it a promising advancement in the field of reinforcement learning.

The M-DQN modification is achieved by adding a scaled log-policy to the DQN's target. This addition introduces a novel term to the DQN's target, similar to maximum-entropy reinforcement learning, and is designed to improve the agent's learning efficiency and overall performance. The M-DQN agent's theoretical aspects include implicit KL regularization, which performs KL regularization without error in the greedy step, and an increase in the action gap, which generalizes advantage learning. These theoretical aspects contribute to the M-DQN agent's superior performance compared to traditional DQN and other state-of-the-art agents.

#### KL regularization

KL regularization, short for Kullback-Leibler regularization, is a technique used in reinforcement learning to penalize a new policy from being too far from the previous one, as measured by the KL divergence. It is a form of regularization that encourages the new policy to stay close to the previous policy, thus preventing drastic changes in the policy distribution. This can be particularly useful in reinforcement learning to ensure stability and prevent large policy changes that may lead to suboptimal behavior. KL regularization has been shown to be helpful in improving the performance and stability of reinforcement learning algorithms, and it is often used in combination with other techniques to achieve better training results.


## Resources

- [Playing Atari with Deep Reinforcement Learning (paper)](https://arxiv.org/abs/1312.5602)
- [Deep Q-Learning Paper Explained (YouTube video)](https://www.youtube.com/watch?v=nOBm4aYEYR4&ab_channel=YannicKilcher)
- [CartPole environment (Github)](https://github.com/openai/gym/blob/master/gym/envs/classic_control/cartpole.py)
- [DQN agent to play SnakeGame(GitHub)](https://github.com/patrickloeber/snake-ai-pytorch/blob/main/snake_game_human.py )
- [DQN agent to play SnakeGame (YouTube video)](https://www.youtube.com/watch?v=L8ypSXwyBds)
- [Deep Q Learning is simple with PyTorch (YouTube video)](https://www.youtube.com/watch?v=wc-FxNENg9U)
- [Wikipedia - Q-learning](https://en.wikipedia.org/wiki/Q-learning)
- [Munchausen RL](https://arxiv.org/abs/2007.14430)
- [Munchausen RL](https://simons.berkeley.edu/sites/default/files/docs/16336/matthieugeistrl20-1slides.pdf)
- [Munchausen RL](https://vitalab.github.io/article/2020/09/10/Munchausen_Reinforcement_Learning.html)

---

**Author:**
Basile Pochet

**Class:**
<Master 2 Data Science for Social Sciences>
