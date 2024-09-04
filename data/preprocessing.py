import json
import re
# from sklearn.model_selection import train_test_split

def preprocess_text(file_path,output_file_path):
    with open(file_path,'r',encoding='utf-8') as file:
        data = file.read()

    #Preprocessing text
     # Remove the Project Gutenberg header and footer
    start_marker = "*** START OF THIS PROJECT GUTENBERG EBOOK HAMLET ***"
    end_marker = "End of the Project Gutenberg EBook of Hamlet, by William Shakespeare"
    
    data = data.split(start_marker)[-1].split(end_marker)[0]
    
    # Normalize the text
    data = data.lower()  # Convert to lowercase
    data = re.sub(r'\s+', ' ', data)  # Replace multiple spaces with a single space
    data = re.sub(r'\[.*?\]', '', data)  # Remove stage directions like [Exit GHOST]
    data = re.sub(r'act \d+', '', data)  # Remove act headers
    data = re.sub(r'scene \d+', '', data)  # Remove scene headers
    data = re.sub(r'[^a-zA-Z\s.,!?]', '', data)  # Remove non-alphabetic characters, except punctuation
    

    with open(output_file_path,'w') as file:
        file.write(data)
    
    char = list(set(data))
    data_size = len(data)
    vocab_size = len(char)

    #Create metadata
    metadata = {
        'data_size' : data_size,
        'vocab_size': vocab_size,
        'characters' : char
    }

    with open('data/config.json', 'w') as json_file:
        json.dump(metadata, json_file, indent=4)



    # Calculate split indices
    train_size = int(0.7 * len(data))
    val_size = int(0.15 * len(data))
    test_size = len(data) - train_size - val_size

    # Sequentially split the data
    train_data = data[:train_size]
    val_data = data[train_size:train_size + val_size]
    test_data = data[train_size + val_size:]

    # Save the splits to files
    with open('data/train_data.txt', 'w') as f:
        f.write(train_data)
    with open('data/val_data.txt', 'w') as f:
        f.write(val_data)
    with open('data/test_data.txt', 'w') as f:
        f.write(test_data)

    
    print(f'The processed text has been saved to {output_file_path}')
    print(f'The config file has been saved to data/config.json')
    print(f'The training text has been saved to data/train_data.txt')
    print(f'The validation text has been saved to data/val_data.txt')
    print(f'The testing text has been saved to data/test_data.txt')

# if __name__ == "__main__":
#     preprocess_text('../data/data.txt','../data/processed.txt')