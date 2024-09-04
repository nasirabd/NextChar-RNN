import numpy as np
import os
import pickle
from tqdm import tqdm

def one_hot_encoding(idx, vocab_size):
    """
    One-hot encode the input index.
    
    :param idx: Index of the character
    :param vocab_size: Size of the vocabulary
    :return: One-hot encoded vector
    """
    vec = np.zeros((vocab_size, 1))
    vec[idx] = 1
    return vec

def save_model(rnn, epoch, save_dir='trained_models'):
    """
    Save the RNN model to a file in the specified directory.
    
    :param rnn: The RNN model instance
    :param epoch: The epoch number to include in the filename
    :param save_dir: The directory where the model will be saved
    """
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    filename = os.path.join(save_dir, f'rnn_model_epoch_{epoch}.pkl')
    with open(filename, 'wb') as f:
        pickle.dump(rnn, f)
    print(f"Model saved to {filename}")



def train_rnn(train_loader, val_loader, rnn, num_epochs=10, learning_rate=0.001, save_interval=None):
    """
    Train the RNN model with training and validation data.
    
    :param train_loader: DataLoader for training data
    :param val_loader: DataLoader for validation data
    :param rnn: The RNN model instance
    :param num_epochs: Number of training epochs
    :param learning_rate: Learning rate for the optimizer
    :param save_interval: Interval for saving the model
    """
    loss_history = []
    for epoch in range(num_epochs):
        train_loader.shuffle()
        epoch_loss = 0.0
        
        # Progress bar for the current epoch with more detailed information
        with tqdm(total=train_loader.get_num_batches(), desc=f"Epoch {epoch + 1}/{num_epochs}", 
                  bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}{postfix}]") as pbar:
            for i in range(train_loader.get_num_batches()):
                batch_inputs, batch_targets = train_loader.get_batch(i)
                # Convert batch inputs to one-hot encoded vectors
                batch_inputs_oh = [one_hot_encoding(idx, train_loader.vocab_size) for idx in batch_inputs.flatten()]
                batch_inputs_oh = np.array(batch_inputs_oh).reshape(train_loader.batch_size, train_loader.seq_length, train_loader.vocab_size)
                
                # Initialize hidden state for each batch
                h_prev = np.zeros((rnn.hidden_size, train_loader.batch_size)) 
                
                # Calculate loss and gradients
                loss, grads = 0, None

                for t in range(train_loader.seq_length):
                    x_t = batch_inputs_oh[:, t, :].T
                    y_t = batch_targets[:,t] 
                    
                    # Forward pass
                    ps, hs = rnn.forward(x_t, h_prev)
                    
                    # Compute loss
                    loss += rnn.loss(ps, y_t)
                    
                    # Backward pass and gradients computation
                    grads = rnn.backward(x_t, hs, h_prev, ps, y_t)
                    # Update h_prev to be used in the next time step
                    h_prev = hs
                    
                    # Update weights
                    rnn.update_parameters(*grads, learning_rate=learning_rate)
                
                epoch_loss += loss
                pbar.set_postfix_str(f"Loss: {loss:.4f}")
                pbar.update(1)  # Update the progress bar
        
        # Track the training loss for each epoch
        train_loss = epoch_loss / train_loader.get_num_batches()
        loss_history.append(train_loss)
        
        # Evaluate on validation set every 100 epochs
        if (epoch + 1) % 10 == 0:
            val_loss, val_accuracy = evaluate_rnn(val_loader, rnn)
            print(f'Epoch {epoch + 1}/{num_epochs}, Train Loss: {train_loss:.4f}, '
                  f'Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_accuracy:.4f}')
        
        # Save the trained model at intervals
        if save_interval is not None and (epoch + 1) % save_interval == 0:
            save_model(rnn, epoch + 1)

    # Optionally, return the loss history for further analysis
    return loss_history



# def train_rnn(data_loader, rnn, num_epochs=10, learning_rate=0.001, save_interval = None):
#     """
#     Training the RNN model to a file in the specified directory.
    
#     :param data_loader: The training data passed as input
#     :param rnn: The RNN model instance
#     :param num_epoch: The number of training steps
#     :param learning_rate: The rate at which the parameter get updated
#     :param save_interval: The model weights saved at a specific interval

#     """
#     loss_history = []
#     for epoch in range(num_epochs):
#         data_loader.shuffle()
#         epoch_loss = 0.0
        
#         for i in range(data_loader.get_num_batches()):
#             batch_inputs, batch_targets = data_loader.get_batch(i)
#             # Convert batch inputs to one-hot encoded vectors
#             batch_inputs_oh = [one_hot_encoding(idx, data_loader.vocab_size) for idx in batch_inputs.flatten()]
#             batch_inputs_oh = np.array(batch_inputs_oh).reshape(data_loader.batch_size, data_loader.seq_length, data_loader.vocab_size)
#             print(f"batch_targets shape: {batch_targets.shape}")
#             # Initialize hidden state for each batch
#             h_prev = np.zeros((rnn.hidden_size, data_loader.batch_size)) 
#             #Caluculating loss and gradients
#             loss, grads = 0, None

#             for t in range(data_loader.seq_length):
#                 x_t = batch_inputs_oh[:, t, :].T
#                 y_t = batch_targets[:, t]
                
#                 # Forward pass
#                 ps, hs = rnn.forward(x_t, h_prev)
               
#                 # Compute loss
#                 loss += rnn.loss(ps, y_t)

#                 # Count correct predictions
#                 predicted_char_idx = np.argmax(ps, axis=0)
#                 correct_predictions += np.sum(predicted_char_idx == y_t)
#                 total_predictions += y_t.shape[0]
                
#                 # Backward pass and gradients computation
#                 grads = rnn.backward(x_t, hs, h_prev, ps, y_t)
#                 # Update h_prev to be used in the next time step
#                 h_prev = hs
                
#                 # Update weights
#                 rnn.update_parameters(*grads, learning_rate=learning_rate)
            
#             epoch_loss += loss
        
#         # Track the loss for each epoch in training
#         loss_history.append(epoch_loss / data_loader.get_num_batches())

#         # Evaluate on validation set
#         val_loss, val_accuracy = evaluate_rnn(val_loader, rnn)
        
#         print(f'Epoch {epoch + 1}/{num_epochs}, Loss: {epoch_loss/data_loader.get_num_batches():.4f}, Accuracy: {accuracy:.4f}')
    
#     #Save the trained model at intervals
#     if save_interval is not None and (epoch + 1) % save_interval == 0:
#             save_model(rnn, epoch + 1)   

def evaluate_rnn(data_loader, rnn):
    total_loss = 0.0
    total_correct = 0
    total_predictions = 0
    
    for i in range(data_loader.get_num_batches()):
        batch_inputs, batch_targets = data_loader.get_batch(i)
        
        # Convert batch inputs to one-hot encoded vectors
        batch_inputs_oh = [one_hot_encoding(idx, data_loader.vocab_size) for idx in batch_inputs.flatten()]
        batch_inputs_oh = np.array(batch_inputs_oh).reshape(data_loader.batch_size, data_loader.seq_length, data_loader.vocab_size)
        
        # Initialize hidden state for each batch
        h_prev = np.zeros((rnn.hidden_size, data_loader.batch_size)) 
        
        batch_loss = 0.0
        correct_predictions = 0
        
        for t in range(data_loader.seq_length):
            x_t = batch_inputs_oh[:, t, :].T
            y_t = batch_targets[:, t]
            
            # Forward pass
            ps, hs = rnn.forward(x_t, h_prev)
            h_prev = hs
            
            # Compute loss
            batch_loss += rnn.loss(ps, y_t)
            
            # Count correct predictions
            predicted_char_idx = np.argmax(ps, axis=0)
            correct_predictions += np.sum(predicted_char_idx == y_t)
        
        total_loss += batch_loss
        total_correct += correct_predictions
        total_predictions += data_loader.seq_length * data_loader.batch_size

    average_loss = total_loss / data_loader.get_num_batches()
    accuracy = total_correct / total_predictions

    return average_loss, accuracy

    
    



