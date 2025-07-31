# CLFs-for-Nonlinear-Robotic-systems
pendulum.py
This Python script simulates an inverted pendulum system, demonstrating a control strategy that combines a Lyapunov-based "bang-bang" controller with a Linear Quadratic Regulator (LQR)-like controller for stabilization around the upright position.

The script visualizes:

The decay of a Lyapunov function, indicating system stability.

The time-evolution of the pendulum's angle (θ) and angular velocity (θ').

An interactive slider allows users to adjust the initial angular velocity ("Initial Kick") to observe its effect on the pendulum's ability to reach and maintain the upright equilibrium. The code identifies and marks the point in time when the system stabilizes within a defined threshold.

Key Features:

Hybrid Control: Implements a switching control strategy.

Lyapunov Stability Analysis: Plots the Lyapunov function to demonstrate stability.

Interactive Simulation: Uses Matplotlib's Slider widget for real-time parameter tuning.

State Visualization: Shows θ and θ' over time.

unicycle.py
This Python script simulates a unicycle robot navigating from a starting position to a goal state using a two-stage analytical controller. The controller aims to first minimize the distance to the goal and then align the robot's orientation with the target orientation.

The simulation visualizes:

The robot's trajectory in the 2D plane.

The evolution of the robot's state variables (x, y, and θ) over time.

The decay of a simple Lyapunov-like function (squared distance error), demonstrating convergence to the goal.

Key Features:

Unicycle Kinematics: Models the differential drive unicycle robot.

Two-Stage Control: Implements a control logic that prioritizes position control then orientation control.

Lyapunov-like Function: Tracks a function to illustrate convergence.

Trajectory Visualization: Plots the robot's path.

State Variable Tracking: Shows how x, y, and θ change during navigation.

Configurable Parameters: Allows easy adjustment of initial/goal states, controller gains, and stopping distance.
