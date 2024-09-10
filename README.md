# Next-Character Prediction RNN

This repository contains code for training a rudementary character-level Recurrent Neural Network (RNN) that generates text based on a given input sequence. The model predicts the next character in a sequence of characters, trained on a text corpus such as Shakespeare or other literature. The codebase is for learning purposes, PyTorch contains a better optimized solution.

## Table of Contents
- [Project Overview](#project-overview)
- [Model Architecture](#model-architecture)
- [Installation](#installation)
- [Usage](#usage)
  - [Data Preparation](#data-preparation)
  - [Training the Model](#training-the-model)
  - [Evaluation](#evaluation)
  - [Generating Text](#generating-text)
  

## Project Overview

This project uses a Recurrent Neural Network (RNN) to learn character-level text generation. The model learns to predict the next character in a sequence based on previous characters. After training, the model can generate text that resembles the style of the input data, such as a Shakespearean play or classic literature.

The code is structured to:
- Preprocess the text data.
- Train an RNN model using a character-level one-hot encoded input.
- Generate new text by predicting the next character in a sequence.

## Model Architecture

The model is a simple RNN with the following structure:
- **Input Layer**: One-hot encoded characters.
- **Hidden Layer**: A recurrent layer (vanilla RNN).
- **Output Layer**: Softmax output to predict the probability distribution of the next character.

**Key Components**:
- **RNN Forward Pass**: Computes hidden states and outputs for each time step.
- **Loss Function**: Cross-entropy loss between the predicted character and the true character.
- **Backward Pass**: Backpropagation through time (BPTT) to calculate gradients and update weights.

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/nasirabd/NextChar-RNN.git


## Usage

## Data Preperation:
1. Download a text corpus, such as Shakespeare or Alice's Adventures in Wonderland, from Project Gutenberg.
2. Place the raw text file in the data/ folder. The file should be named raw.txt.

## Training the model:
Run the script in the terminal.
```bash
python main.py

## Evaluation:
The model will also automatically evaluate after main.py is called.

## Generating Text:
After the training the model automatically generates a new text created from learned model weights.

## Hyperparameters 
You can modify the hyperparameters in main.py to control the training:
- hidden_size: Number of hidden units in the RNN.
- seq_length: The length of the input character sequence.
- learning_rate: Learning rate for optimization.
- num_epochs: Number of training epochs.
- save_interval: Number of epochs between model saves.


