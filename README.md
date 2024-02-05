# RL_flappy_bird

## Overview
Flappy Bird is a game in which the player controls a bird attempting to fly between columns of green pipes without hitting them
<br>
This project implements Reinforcement Learning (RL) algorithms to train a model to play Flappy Bird game. The code is designed to provide a demonstration of Q-learning in action, showcasing how an agent learns to navigate and improve its performance in the Flappy Bird environment.

## Table of Contents

- [Introduction](#introduction)
- [Installation](#installation)
- [Usage](#usage)
- [Algorithm Details](#algorithm-details)
- [Customization](#customization)

## Introduction

The `SmartFlappyBird` class is the main component of this project, incorporating the Q-learning algorithm to train an agent to play Flappy Bird. The game environment is provided by the `flappy_bird_gym` library, and the core RL functions are implemented in the main file.

## Installation

To run the project, follow these steps:

1. Clone the repository:

   ```bash
   git clone https://github.com/mobinbr/RL_flappy_bird
2. Navigate to the project directory:

    ```bash
    cd RL_flappy_bird
> I suggest creating a virtual environment before proceeding with the following steps to mitigate potential issues.

3. Install the required dependencies:

    ```bash
    pip install -r requirements.txt
## Usage 
Run the SmartFlappyBird class by executing the main file:

    python main.py

This will initiate the training process for the specified number of iterations; wait for a few seconds (30 seconds or more for training)
<br>
you can speed up the program by lowering the FPS.
<br>
After training, the model will run in the Flappy Bird environment to demonstrate its learned behavior.

## Algorithm Details
The Q-learning algorithm is implemented in the `SmartFlappyBird` class. Key components include:

- **Q-values:** The Q-values are stored in a Counter, and the Q-learning formula is used to update these values during training.

- **Policy Function:** The `policy` method determines the action to be taken based on the current state, either by exploiting the learned Q-values or by choosing a random action with a certain probability.

- **Reward Function:** The `compute_reward` method calculates the reward for a given action, encouraging the agent to learn optimal strategies for maximizing its score.

- **Training Loop:** The `run_with_policy` method is responsible for training the model using the Q-learning algorithm. The `run_with_no_policy` method demonstrates the model's performance after training.

## Customization
Adjust the parameters in the SmartFlappyBird class and the bird speed to customize the training process. Experiment with hyperparameters such as epsilon, alpha, and lambda to observe their impact on the agent's learning.