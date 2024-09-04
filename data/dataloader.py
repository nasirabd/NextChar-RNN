import numpy as np
import json

class TextLoader:
    #Initializing the class constructor and attributes
    def __init__(self, data_path='../data/processed.txt', seq_length=None, batch_size=None, config_path='..data\config.json'):
    
            #Reading the preprocessed text and it's config file
            with open(data_path, 'r') as f:
                self.data = f.read()
            # print(f"Loaded data: {self.data[:10]}")

            with open(config_path, 'r') as f:
                config = json.load(f)

    
            self.data_size = config['data_size']
            self.vocab_size = config['vocab_size']
            # print(f'the vocab size is : {self.vocab_size}')
            self.characters = config['characters']
            self.seq_length = seq_length if seq_length is not None else 5
            self.batch_size = batch_size if batch_size is not None else 4

            # print(f"Initialized TextLoader with batch_size={self.batch_size}, seq_length={self.seq_length}")

            #Creating character mappings
            #This will give our model a consistent numerical representation
            self.char2idx = {char:idx for idx,char in enumerate(self.characters)}
            #This will help decode our predictions back to characters
            self.idx2char = {idx:char for idx,char in enumerate(self.characters)}

            #Prepare the data for training with input and target
            self.input_seqs, self.target_chars = self.create_seq()
            self.num_batches = len(self.input_seqs) // self.batch_size 
            # print(f"Total sequences: {len(self.input_seqs)}, Total batches: {self.num_batches}")

    def create_seq(self):
        input_seqs = []
        target_chars = []
        for i in range(0, len(self.data) - self.seq_length):
            input_seq = self.data[i: i + self.seq_length]
            target_char = self.data[i + 1: i + 1 + self.seq_length]
            input_seqs.append([self.char2idx[char] for char in input_seq])
            target_chars.append([self.char2idx[char] for char in target_char])  # Notice the list here to make target_chars 2D
        return np.array(input_seqs), np.array(target_chars)
    
    def shuffle(self):
        combined = list(zip(self.input_seqs,self.target_chars))
        np.random.shuffle(combined)
        self.input_seqs[:],self.target_chars[:] = zip(*combined)
        return self.input_seqs, self.target_chars

    def get_batch(self, batch_index):
        start = batch_index * self.batch_size
        end = start + self.batch_size
        # print(f"Batch {batch_index+1}: start={start}, end={end}, total sequences={len(self.input_seqs)}")
        batch_inputs = np.array(self.input_seqs[start:end])
        batch_targets = np.array(self.target_chars[start:end])    
        return batch_inputs, batch_targets

    def get_num_batches(self):
        return self.num_batches

    def get_char_to_ix(self):
        return self.char_to_ix

    def get_ix_to_char(self):
        return self.ix_to_char
        



    
