# ddpg_helicopter

This repository contains the code developed as part of the assignment for AE4350 Bio-Inspired Intelligence and Learning for Aerospace Applications. More information about the helicopter model is available [here](https://santiago-valencia.com/posts/helicopter-simulation/helicopter_model.html).


# Scripts

The core functionality of the developed code is split in three scripts:
- `helicopter_model.py` contains the 3-degree-of-freedom helicopter simulation model that determines the dynamics of the system to be controlled via reinforcement learning (based on previous own work for the course AE4314 Helicopter Performance, Stability, and Control).
- `helicopter_environment.py` contains the `HelicopterTrain` class, an OpenAI Gym environment subclass for training an agent to control the helicopter.
- `training_utils.py` contains the implementation of the DDPG algorithm (based on [this Keras tutorial by Hemant Singh](https://keras.io/examples/rl/ddpg_pendulum/)).

Other files in the repository include a sample training run in `training_run.py`, sensitivity analyses in scripts ending with `_sensitivity.py`, and a Jupyter notebook with a comparison of the different agents developed.

# Requirements
The code shown here was run in an environment with Python 3.9 and TensorFlow 2.9. The full requirements are shown in the attached `requirements.txt`.
