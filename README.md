# SideProject_RL-Driver
Experiments with different methods of reinforcement learning for a self-driving car in a donkeycar simulator.

Original ddqn.py code is from https://github.com/tawnkramer/gym-donkeycar/tree/master/examples/reinforcement_learning, which implements a naive double-deep q-learning algorithm to be used as a starting point to see how different machine learning and image processing techniques can increase the efficacy of the reinforcement learner.

First, I trained an autoencoder on Canny edge-detected frames recorded from the simulator in varied situations, on and off the track. This autoencoder learns how best to encode an 80x80 pixel black and white image into just 20 numbers, allowing the q-learning algorithm to improve much faster.
While the car is running in the simulator, each frame from the front camera is passed through Canny edge-detection and then through the autoencoder. Thus the algorithm stores its state in a much less data-intensive way, mitigating the curse of dimensionality.

The original algorithm must train for about 24 hours before it can occasionally make it around an entire lap. With these improvements, the self driving car can consistently make it around the entire track after an hour. This repository includes small videos of the improved car simulator at the start of training, and after 55 minutes of training have passed. In the videos, the top left window shows the car's camera view after edge detection, and the top right window shows the camera view after being extremely compressed and decompressed using the autoencoder.
