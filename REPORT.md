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

Describe the architecture of a typical Deep Q-Learning network.

### Training Process

Detail the training process of a Deep Q-Learning model. 

# Munchausen agent

Basically, it's a DQN agent where the exploratory part is more important and rewarded. 

## Resources

- [Playing Atari with Deep Reinforcement Learning (paper)](https://arxiv.org/abs/1312.5602)
- [Deep Q-Learning Paper Explained (YouTube video)](https://www.youtube.com/watch?v=nOBm4aYEYR4&ab_channel=YannicKilcher)
- [CartPole environment (Github)](https://github.com/openai/gym/blob/master/gym/envs/classic_control/cartpole.py)
- [DQN agent to play SnakeGame(GitHub)](https://github.com/patrickloeber/snake-ai-pytorch/blob/main/snake_game_human.py )
- [DQN agent to play SnakeGame (YouTube video)](https://www.youtube.com/watch?v=L8ypSXwyBds)
- [Deep Q Learning is simple with PyTorch (YouTube video)](https://www.youtube.com/watch?v=wc-FxNENg9U)
- [Wikipedia - Q-learning](https://en.wikipedia.org/wiki/Q-learning)

## Conclusion

Conclusion

## Acknowledgments


---

**Author:**
Basile Pochet

**Class:**
<Master 2 Data Science for Social Sciences>
