# Digit-Recognition-Gui
This project is a graphical user interface (GUI) application that predicts handwritten digits using a neural network trained on the MNIST dataset. The project demonstrates the impact of noise on digit recognition and includes functionality for users to draw digits, apply noise, and receive predictions.

Features : 
1. GUI for Digit Drawing: Users can draw digits on a canvas and get predictions instantly.
2. MNIST Neural Network: The model is trained on the MNIST dataset, ensuring high accuracy for handwritten digits.
3. Noise Handling: Demonstrates the model's robustness by allowing users to add noise to the input image.
4. Real-time Predictions: Instant feedback with probability distributions for digit predictions.

Dataset:
This project uses the MNIST dataset of handwritten digits for training and testing. To make the model more robust, additional noise is applied to the training images during preprocessing.

How It Works:
Train the Neural Network:
Load the MNIST dataset.
Apply Gaussian noise to the training images.
Train a neural network (e.g., a simple CNN) on the noisy dataset.

GUI Features:
Users draw a digit in a canvas area.
The digit can be "noised" by the user to simulate real-world conditions.
The model predicts the digit and displays the confidence levels.
Tech Stack
Python
Tkinter: For building the GUI application.
TensorFlow/Keras: For training the neural network model.
NumPy: For data preprocessing and noise addition
