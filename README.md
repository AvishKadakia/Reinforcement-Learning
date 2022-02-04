# Reinforcement-Learning
Read Me for Project 1
I have used the AI-Gym environment provided by OpenAI to test 2 learning and control algorithms.

Method 1 (DQN):
The first method uses q learning to compute loss on the basis of reward gained. A neural network is
used to make the prediction and it is updated using Markov decision process (MDP). The neural network
improves by training on the replays of previous experiences over time.

Method 2 (ANN with GA):
The second method uses genetic algorithms to optimize the weights of a neural network over multiple
generations. It starts by creating a random population. Each individual is then given a fitness score which
the time for which they have survived in the game. Then the most fit individuals are chosen to create off
springs for the next generation. This process is then continued over several generations until the desired
score is achieved.
