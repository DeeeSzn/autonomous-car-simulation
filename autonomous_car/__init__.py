"""
Autonomous Car Simulation + Controller
=======================================

A lightweight simulation for testing autonomous driving controllers using:
- Locally Weighted Regression (LWR) for supervised policy approximation
- Reinforcement Learning (SAC/TD3) for policy improvement

Modules:
    env: Gymnasium environment with kinematic bicycle model
    controllers: Expert (pure pursuit), LWR, and RL-based controllers
    utils: Track geometry, data collection, evaluation metrics
"""

__version__ = "0.1.0"
