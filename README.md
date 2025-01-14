# Reinforcement Learning for Autonomous Surface Vessels
A project exploring autonomous surface vessel navigation and collision avoidance using reinforcement learning.

## Gymnasium Environment

The environment is a custom Gymnasium environment that simulates the ASV navigation task.  This environment simulates vessel movements using an action space that controls acceleration and rudder position.

## Training

[Train](rc/sac/train.py) contains code that trains a Soft Actor-Critic model to learn how to navigate to the target quickly and avoid obstacles.

## Testing

[Testing](src/test_trained_model.py) simulates the model on a new randomly initialized environment based on the model file from specific weights and biases run-id.
