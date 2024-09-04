import matplotlib.pyplot as plt
import numpy as np
from tools import one_hot_encoding

def lossgraph(num_epochs,loss_history):
    """
    Generate plot using the loss from training.
    
    :param num_epochs: The number of epochs while training 
    :param loss_history: The loss history from each training epoch
    :return: Generated plot function of the loss vs epoch
    """
    plt.plot(range(1, num_epochs + 1), loss_history, label='Training Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training Loss Curve')
    plt.legend()
    # plt.show()
    plt.savefig('loss_plot.png')
    

def generate_text(rnn, data_loader, start_char='a', length=100):
    """
    Generate text using the trained RNN model.
    
    :param rnn: The trained RNN model
    :param data_loader: The data loader containing character mappings
    :param start_char: The initial character to start the sequence
    :param length: The length of the sequence to generate
    :return: Generated text as a string
    """
    # Initialize hidden state
    h_prev = np.zeros((rnn.hidden_size, 1))
    
    # Get the index of the start character
    start_idx = data_loader.char2idx[start_char]
    
    # One-hot encode the start character
    x = one_hot_encoding(start_idx, data_loader.vocab_size)
    
    # Initialize the output text with the start character
    generated_text = start_char
    
    for _ in range(length):
        # Perform the forward pass
        ps, h_prev = rnn.forward(x, h_prev)
        
        # Sample from the probability distribution
        next_idx = np.random.choice(range(data_loader.vocab_size), p=ps.ravel())
        
        # Convert index to character
        next_char = data_loader.idx2char[next_idx]
        
        # Append the character to the generated text
        generated_text += next_char
        
        # Update the input to the next character's one-hot encoding
        x = one_hot_encoding(next_idx, data_loader.vocab_size)
    
    return generated_text
