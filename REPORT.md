# Deep Q-Learning

## Introduction

If we want to talk about Deep Q-Learning we need to introduce Reinforcement Learning, Deep Q-Learning being a type of RL.
Reinforcement learning is a type of machine learning that trains algorithms to learn from their environments through trial and error. To do that, an agent is given information about its environment and do an action accordingly. It receives then a reward or a punishment so it can optimize its behaviors over time. The data is ccumulated through the agent's interactions with the environment. In other words, "reinforcement learning is teaching a software agent how to behave in an environment by telling it how good it's doing" ([Patrick Loeber](https://www.youtube.com/watch?v=L8ypSXwyBds).
Reinforcement learning is being used in various domains such as game playing, autonomous systems, robotics etc and is a powerful approach for training intelligent agents.

## Deep Q-Learning

Deep Q-Learning is a type of reinforcement learning where we use a deep neural network to predict the actions. 

### Key Concepts

Q value is standing for quality of action, it represents the reward we obtain if we play this action while being at this state. 

The basic main loop we have to understand for DQL is the following (after initializing the Q-value): 

- choose the action (from the state)
- perform the action
- measure the reward
- update Q-value and train the model.

The last step, training the model, needs a loss function. It is the reason why we use the **Bellman Equation**

\[ Q(s, a) = (1 - \alpha) \cdot Q(s, a) + \alpha \cdot \left(r + \gamma \cdot \max_a Q(s', a)\right) \]

Summarize the key concepts of Deep Q-Learning, including:

- Q-Learning
- Experience Replay
- Target Networks
- Exploration vs. Exploitation (Epsilon-Greedy)

### Architecture

Describe the architecture of a typical Deep Q-Learning network. Discuss the role of neural networks, input and output representations, and the learning process.

### Training Process

Detail the training process of a Deep Q-Learning model. Explain how the model learns from experiences, updates its parameters, and improves its performance over time.

# Munchausen agent

Basically, it's a DQN agent where the exploratory part is more important and rewarded. 

## Resources

- [Playing Atari with Deep Reinforcement Learning (paper)](https://arxiv.org/abs/1312.5602)
- [Deep Q-Learning Paper Explained (YouTube video)](https://www.youtube.com/watch?v=nOBm4aYEYR4&ab_channel=YannicKilcher)
- [CartPole environment (Github)](https://github.com/openai/gym/blob/master/gym/envs/classic_control/cartpole.py)
- [DQN agent to play SnakeGame(GitHub)](https://github.com/patrickloeber/snake-ai-pytorch/blob/main/snake_game_human.py )
- [DQN agent to play SnakeGame (YouTube video)](https://www.youtube.com/watch?v=L8ypSXwyBds)
- [Deep Q Learning is simple with PyTorch (YouTube video)](https://www.youtube.com/watch?v=wc-FxNENg9U) 

## Conclusion

Conclusion

## Acknowledgments


---

**Author:**
Basile Pochet

**Class:**
<Master 2 Data Science for Social Sciences>
