from data import preprocess_text, TextLoader
from model import RNN
from tools import train_rnn, one_hot_encoding, evaluate_rnn
from visualization import lossgraph, generate_text
# from sklearn.model_selection import train_test_split

if __name__ == "__main__":
    # Preprocess the data, this saves the train, val, and testing splits
    preprocess_text('data/raw.txt', 'data/processed.txt')
    
    
    # Load the data for each split
    train_loader = TextLoader(data_path='data/train_data.txt', seq_length=5, batch_size=1, config_path='data/config.json')
    val_loader = TextLoader(data_path='data/val_data.txt', seq_length=5, batch_size=1, config_path='data/config.json')
    test_loader = TextLoader(data_path='data/test_data.txt', seq_length=5, batch_size=1, config_path='data/config.json')

    
    # Step 4: Initialize the RNN model
    input_size = train_loader.vocab_size
    hidden_size = 100  # Number of hidden units
    output_size = train_loader.vocab_size
    
    rnn = RNN(input_size, hidden_size, output_size)
    
    # Train the RNN model
    loss_history = train_rnn(train_loader, val_loader, rnn, num_epochs=100, learning_rate=0.001, save_interval = 10)

    # Visualize the training loss 
    lossgraph(num_epochs = 100, loss_history = loss_history)

    # Evaluate the model on the test set
    test_loss,test_accuracy = evaluate_rnn(test_loader, rnn)
    print(f'Test Loss: {test_loss:.4f}',f'Test Accuracy: {test_accuracy:.4f}')


    start_char = 't'  
    generated_text = generate_text(rnn, train_loader, start_char=start_char, length=200)
    print(f'\nGenerated Text:\n{generated_text}')