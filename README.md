# better-faster-stronger
*Completed in collaboration between Tej Shah, Shivam Agrawal, & Srinandini Marpaka*

## Overview
Using the same [Circle of Life](https://github.com/tejpshah/circle-of-life) environment from Project 2, we build a few intelligent agents for different environments. In the complete information environment, we use dynamic programming and value iteration to iteratively solve for the Bellman Equations to calculate optimal utilities for every state in the state space. Using those utilities U* we build an agent with the optimal policy using the Bellman Equations formulation. For data efficiency, we develop and implement a neural network function approximator V from scratch to approximate U* which can be queried at inference time for making decisions. In the partial prey information setting, we develop $U_{partial}$ which is the expected value of all the optimal utilities ${U^*}$ for every location of prey using our probability distribution $p_{prey}$ For data efficiency, we develop and implement a generalized neural network function approximator $V_{partial}$ to predict $U_{partial}$ from data. Finally, we outline an optimal $U^{*}_{partial}$ using the temporal difference method of Deep-Q Learning as well as an alternative of Value Iteration with Neural Networks. 