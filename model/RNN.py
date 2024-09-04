import numpy as np

class RNN:
    def __init__(self,input_size, hidden_size, output_size):
        #Define the input, output, and hidden size
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        # Initialize the wieghts for linearization
        #weights from prev hidden to hidden
        self.Whh = np.random.rand(hidden_size, hidden_size) * .01
        #weights from input to hidden
        self.Wxh = np.random.rand(hidden_size, input_size) * .01
        #weights from hidden to output
        self.Why = np.random.rand(output_size, hidden_size) * .01

        #Bias terms
        self.bh = np.zeros((hidden_size,1))
        self.by = np.zeros((output_size,1))

    def forward(self,input, h_prev):
        """
        Perform the forward pass of the RNN.
        
        :param inputs: One-hot encoded inputs (input_size, sequence_length)
        :param h_prev: Previous hidden state (hidden_size, 1)
        :return: Output probabilities, final hidden state
        """
        # hs, ys, ps = {}, {}, {}
        # hs[-1] = h_prev
        # for t in range(len(inputs)):
        #     # Compute hidden state
        hs = np.tanh(np.dot(self.Wxh, input) + np.dot(self.Whh, h_prev) + self.bh)
        # Compute output logits
        ys = np.dot(self.Why, hs) + self.by
        # Apply softmax to get output probabilities
        ps = np.exp(ys) / np.sum(np.exp(ys), axis=0, keepdims=True)
        
        return ps, hs
    
    def loss(self, ps, target):
        """
        Compute the loss (cross-entropy loss).
        
        :param ps: Output probabilities from the forward pass
        :param targets: True indices of the target characters
        :return: Cross-entropy loss
        """
        loss = 0
        # for t in range(len(targets)):
        loss = -np.sum(np.log(ps[target, np.arange(target.shape[0])]))
        return loss
    
    def backward(self, input, hs, h_prev, ps, target):
        """
        Perform the backward pass of the RNN (BPTT).
        
        :param inputs: One-hot encoded inputs (input_size, sequence_length)
        :param hs: Hidden states from the forward pass
        :param ps: Output probabilities from the forward pass
        :param targets: True indices of the target characters
        :return: Gradients of the weights and biases
        """
        # Initialize gradients
        dWxh, dWhh, dWhy = np.zeros_like(self.Wxh), np.zeros_like(self.Whh), np.zeros_like(self.Why)
        dbh, dby = np.zeros_like(self.bh), np.zeros_like(self.by)
        dh_next = np.zeros_like(hs[0])
        
        # for t in reversed(range(len(targets))):
        dy = np.copy(ps)
        dy[target, np.arange(target.shape[0])] -= 1 
        # dy[target] -= 1  # Compute the gradient for y
        dWhy += np.dot(dy, hs.T)
        dby += dy
        dh = np.dot(self.Why.T, dy) + dh_next  # Backprop into h
        dhraw = (1 - hs ** 2) * dh  # Backprop through tanh
        dbh += dhraw
        dWxh += np.dot(dhraw, input.T)
        dWhh += np.dot(dhraw, h_prev.T)
        dh_next = np.dot(self.Whh.T, dhraw)
        
        # Clip gradients to avoid exploding gradients
        for dparam in [dWxh, dWhh, dWhy, dbh, dby]:
            np.clip(dparam, -5, 5, out=dparam)
        
        return dWxh, dWhh, dWhy, dbh, dby
    
    def update_parameters(self, dWxh, dWhh, dWhy, dbh, dby, learning_rate=0.001):
        """
        Update the parameters using gradient descent.
        
        :param dWxh: Gradient of Wxh
        :param dWhh: Gradient of Whh
        :param dWhy: Gradient of Why
        :param dbh: Gradient of bh
        :param dby: Gradient of by
        :param learning_rate: Learning rate for gradient descent
        """
        self.Wxh -= learning_rate * dWxh
        self.Whh -= learning_rate * dWhh
        self.Why -= learning_rate * dWhy
        self.bh -= learning_rate * dbh
        self.by -= learning_rate * dby