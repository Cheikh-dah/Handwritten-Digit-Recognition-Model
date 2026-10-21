# Handwritten Digit Recognition Model

We're going to build a neural network that recognizes handwritten numbers (0–9) using Python, NumPy, Matplotlib, and Jupyter Notebook.

No PyTorch. No TensorFlow. No pretrained models.

## What we're building

![Our neural network layers](pic/our-neurals-net-layers.png)

Our neural network will have three layers:

- **Input layer:** 784 neurons, one for each pixel of a 28×28 image.
- **Hidden layer:** 128 neurons that learn patterns in the images.
- **Output layer:** 10 neurons, one for each digit from 0 to 9.

## Stack

- Python
- NumPy
- Matplotlib
- Jupyter Notebook

## Project structure

```
Handwritten-Digit-Recognition-Model/
├── app.py
├── data/          # MNIST dataset
├── models/        # Trained model weights and saved model
├── notebooks/     # Jupyter notebooks
├── pic/           # Architecture diagrams
│   └── our-neurals-net-layers.png
├── requirements.txt
└── README.md
```
