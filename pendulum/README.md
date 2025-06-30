# Swing-Up and LQR Control Simulation of an Inverted Pendulum

This project simulates the swing-up and LQR control stabilisation of an inverted pendulum on a cart using a Python implemented physics engine and MATLAB based controller design. It demonstrates how to transition from an energy-based swing-up strategy to precise stabilisation using Linear Quadratic Regulation (LQR).

![Portfolio Gif](https://github.com/fcracknell/portfolio/blob/main/files/pendulum.gif)

---

# Project Overview

- **Languages:** Python & MATLAB    
- **Control Strategy:**
  - **Swing-Up Phase:** Energy-based control using system potential & kinetic energy ([1], [2])
  - **Inverted stabilisation:** LQR (Linear Quadratic Regulator) based on a linearized system ([3], [4])
- **Extras:** Real-time simulation, pendulum trail visualisation, gif recording, data plots

---

# Dependencies

## Python

- numpy
- pygame
- pymunk
- matplotlib
- pygame-screen-recorder (optional: only for gif recording)

## MATLAB

- Control System Toolbox

# Outputs

- Real-time simulation of cart-pendulum swing-up and inverted control 
- Time-series plots of:
  - Cart position and velocity
  - Pendulum angle and angular velocity
  - Swing-up force input
  - Normalised mechanical energy
  - LQR total force and decomposition by state variable

# References
- [1] Energy-Based Swing-Up: https://coecsl.ece.illinois.edu/se420/ast_fur96.pdf
- [2] Energy-Based Swing-Up: https://youtu.be/RhF2NMCYoiw
- [3] LQR Control: https://ctms.engin.umich.edu/CTMS/index.php?example=InvertedPendulum&section=SystemModeling
- [4] LQR Control: https://www.youtube.com/watch?v=96hHEWN1sIM
