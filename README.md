MNIST Digit Recognition from Scratch

This project implements a digit recognition model for the MNIST dataset using only NumPy and Pandas, without any deep learning frameworks like TensorFlow or PyTorch. The model is built entirely from scratch, showcasing core machine learning concepts such as forward propagation, backpropagation, and gradient descent.

📂 Project Structure

train.csv: The training dataset containing images of handwritten digits (0-9).

main.py: Contains the implementation of the digit recognition model.

🚀 Model Overview

Data Preprocessing: The data is loaded using Pandas and normalized.

Neural Network Architecture:

Two-layer neural network

Input Layer: 784 neurons (28x28 flattened images)

Hidden Layer: 10 neurons with ReLU activation

Output Layer: 10 neurons with Softmax activation

Training:

Implemented forward propagation, backward propagation, and parameter updates using gradient descent.

Prediction & Evaluation: Displays prediction accuracy and visualizes predictions using Matplotlib.

⚙️ Setup Instructions

git clone <repository_url>
cd mnist-digit-recognition
pip install -r requirements.txt
python main.py

📊 Results

The model outputs the accuracy of predictions during training and visualizes example predictions with their actual labels.

🛠️ Technologies Used

Python

NumPy

Pandas

Matplotlib

🎯 Future Improvements

Implement additional hidden layers for deeper learning.

Introduce learning rate scheduling and regularization.

Experiment with different activation functions.

📄 License

This project is licensed under the MIT License.
