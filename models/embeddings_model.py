# Word Embeddings

'''
This class has a function called "get_embeddings" in the
end which takes an input file and generates the embeddings
for each sentence and saves it in a file

'''
# 0. Imports

# importing all necessary modules
import pandas as pd
import torch
from transformers import XLNetTokenizer, XLNetModel, BertTokenizer, BertModel
import os

# needs to be run only once in the beginning

# nltk.download('punkt')

### Function Definations

def read_file(file_name):
    '''
    This function reads the file and returns it's content in a pandas framework

    - Inputs:
        file_name (str): path to the file to be read

    - Output:
        data (pandas framework): pandas framework containing the data

    '''
    data = pd.read_csv(file_name, sep=',')
    return data


def preprocess(text_column, model_name=None):
    '''
    Takes in a pandas column containing text and preprocesses it.
    Also creates segment ids if it is required by the model

    - Inputs:
        text_column (pandas series): The column containing all text entries to be processed
        model_name (str): name of the model to be used for generating embeddings

    - Output:
        preprocessed_indexes_column (list): a list of all text tokenized, encoded and padded.

    '''

    # Load pre-trained model tokenizer (vocabulary)
    if model_name == 'bert':
        tokenizer = BertTokenizer.from_pretrained('bert-large-uncased')
        pad_id = 0
    elif model_name == 'xlnet':
        tokenizer = XLNetTokenizer.from_pretrained('xlnet-large-cased')
        pad_id = tokenizer.pad_token_id
    else:
        print('Please Specify the appropriate model name.')

    # convert the pandas series to a list for the sake of easier processing
    text_column = list(text_column)

    # define place holders
    preprocessed_indexes_column = []

    # iterate through each sentence in the list
    max_length = 0
    for text in text_column:
        # iterate through each sentence in the paragraph, only one sentence in each paragraph

        if model_name == 'bert':
            text = "[CLS] " + text + " [SEP]"
        elif model_name == 'xlnet':
            text = text + " [SEP] [CLS]"
        else:
            text = text

        # tokenize the sentence into words using the model specific tokenizer
        tokenized_text = tokenizer.tokenize(text)

        # Map the token strings to their vocabulary indices.
        indexed_tokens = tokenizer.encode(tokenized_text, add_special_tokens=False)
        preprocessed_indexes_column.append(indexed_tokens)

        # keeping track of the maximum sequence length
        max_length = len(tokenized_text) if len(tokenized_text) > max_length else max_length

    # pad the sequence to allow batch processing
    preprocessed_indexes_column = pad_sequences(preprocessed_indexes_column, max_length, pad_id)
    return preprocessed_indexes_column


def pad_sequences(data, max_length, pad_id):
    '''
    Pads all the  sequence in 'data' with 'pad_id' to have a constant length of 'max_length'
    - Inputs:
        text_column (pandas series): The column containing all text entries to be processed
        model_name (str): name of the model to be used for generating embeddings

    - Output:
        preprocessed_indexes_column (list): a list of all text tokenized, encoded and padded.

    '''
    data_paded = []
    for d in data:
        if len(d) < max_length:
            temp = d + [pad_id] * (max_length - len(d))
            data_paded.append(temp)
        else:
            data_paded.append(d)
    return data_paded


def generate_embedding(paragraphs, model_name):
    '''
    Takes in a pandas dataframe and generates embeddings for the text column
    - Inputs:
        pd_dataframe (pandas dataframe): The dataframe containing all text column and their ids
        model_name (str): name of the model to be used for generating embeddings

    - Output:
        sentence_embedding (tensor): tensor of shape n by 1024 where n is the number of sentence

    '''
    # Convert inputs to PyTorch tensors
    tokens_tensor = torch.tensor([paragraphs]).squeeze()

    if model_name == 'bert':
        # Load pre-trained bert model (weights)
        model = BertModel.from_pretrained('bert-large-uncased')
    elif model_name == 'xlnet':
        # Load pre-trained xlnet model (weights)
        model = XLNetModel.from_pretrained('xlnet-large-cased')

    # Put the model in "evaluation" mode, meaning feed-forward operation.
    model.eval()

    # Predict hidden states features for each layer
    with torch.no_grad():
        encoded_layers = model(tokens_tensor)

    # taking embeddings of the last layer.
    # token_vecs` is a tensor with shape [n x k x 768]
    token_vecs = encoded_layers[0]

    # Calculate the average of all k  token vectors.
    sentence_embedding = torch.mean(token_vecs, dim=1)

    print("Our final sentence embedding vector of shape:", sentence_embedding.size())
    return (sentence_embedding)


def get_embeddings(model_name, file_name =None, sentences= None):
    '''
    wrapper functions which generates the the embeddings for the text in input file and saves in a json file
    - Inputs:
        file_name (str): string pointing to the file to be read
        model_name (str): name of the model to be used for generating embeddings

    - Output:
        sentence_embedding (tensor): tensor of shape n by 1024 where n is the number of sentence

    '''


    # Read file if file path is provided
    if file_name is not None:
        pd_data = read_file(file_name)
    elif sentences is not None:
        pd_data = pd.DataFrame()
        pd_data['paragraph'] = sentences
    else:
        print ('Either a file name or a list of sentences must be provided')
        return


    # do preprocessing on text
    pd_data['paragraph'] = preprocess(pd_data['paragraph'], model_name)

    # generate embeddings
    sentence_embedding = generate_embedding(pd_data['paragraph'], model_name).tolist()

    # save embeddings
    import json
    output_folder = 'output/embeddings/'
    file_name = output_folder + model_name + '_embeddings.json'
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    with open(file_name, 'w') as fp:
        json.dump(sentence_embedding, fp)

    return sentence_embedding