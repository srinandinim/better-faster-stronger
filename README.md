# Graduate AI Project 3: Better, Faster, Stronger
This project involves building a simulator for the "Circle-Of-Life" game and developing intelligent "learning" agents to play the game in complete information and partial prey information settings. The project was completed in collaboration between Tej Shah, Srinandini Marpaka, & Shivam Agrawal.

## Links

Assignment: https://github.com/srinandinim/better-faster-stronger/blob/master/labreport/assignment3_description.pdf

Lab Report: https://github.com/srinandinim/better-faster-stronger/blob/master/labreport/assignment3_labreport.pdf

## Motivation
How can we design intelligent agents that can learn from experience and improve their performance over time? How do we represent and encode knowledge in a form that can be used to make predictions and decisions? How do we train machine learning algorithms to recognize patterns and make accurate predictions? How do we evaluate and compare different learning algorithms, and how do we select the best one for a given task? These are just some of the questions that we explore in this module on learning.

Through this project, we delve into a wide range of techniques and approaches for building intelligent agents that can learn from data. In this module, we study both supervised and unsupervised learning algorithms, and examine how these methods can be used to make predictions, classify data, cluster data, and learn from reinforcement feedback. We also cover more advanced topics such as deep learning, convolutional neural networks, and generative adversarial networks. By studying these topics, we gain a deeper understanding of how to design intelligent agents that can adapt and improve their performance through experience.

## The Simulator
We use the [Circle of Life](https://github.com/tejpshah/circle-of-life) environment from Project 2 as a testbed for developing and evaluating learning algorithms. In this environment, an agent must navigate a graph and try to catch the prey before being caught by the predator. The environment has different levels of information available to the agent, ranging from complete information to partial prey information. The main difference in Project 3 from Project 2 is that the predator is always easily distracted, meaning that it moves optimally with 0.6 probability and randomly otherwise.

## The Agents
In the complete information environment, we use dynamic programming and value iteration to iteratively solve for the Bellman Equations to calculate optimal 
utilities for every state in the state space. Using those utilities U* we build an agent with the optimal policy using the Bellman Equations formulation. For data efficiency, we develop and implement a neural network function approximator V from scratch to approximate U* which can be queried at inference time for making decisions. In the partial prey information setting, we develop $U_{partial}$ which is the expected value of all the optimal utilities U* for every location of prey using our probability distribution $p_{prey}$ For data efficiency, we develop and implement a generalized neural network function approximator $V_{partial}$ to predict $U_{partial}$ from data. Finally, we outline an optimal $U^{*}_{partial}$ using the temporal difference method of Deep-Q Learning as well as an alternative of Value Iteration with Neural Networks. 

## Experimental Results
For detailed results, we direct interested readers to the lab report. All the "learning" agents developed in this project achieve 100% success rate for every information setting. When we use value iteration, all agents move with the fewest number of steps taken, with little variance. When we replace the optimal utilities with a function approximator neural network for the utilities, we have more variance with respect to the number of steps taken. Generally, as more uncertainty increases, the variance in the number of steps also increases. 

# Academic Integrity
Please follow both Rutgers University's Principles of Academic Integrity and the Rutgers Department of Computer Science's Academic Integrity Policy.


