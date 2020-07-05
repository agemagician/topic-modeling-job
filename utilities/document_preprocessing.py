import re
import nltk
from nltk.stem.porter import PorterStemmer
from stop_words import get_stop_words
from nltk.tokenize import word_tokenize
import pandas as pd
import numpy as np
from langdetect import detect

#downloads
#nltk.download('crubadan')

# sentence level cleaning
def preprocess_sent(s):
    """
    :param s: string to be processed
    :return: processed string: see comments in the source code for more info
    """


    # normalization 1: xxxThis is a --> xxx. This is a (missing delimiter)
    s = re.sub(r'([a-z])([A-Z])', r'\1\. \2', s)  # before lower case
    # normalization 2: lower case
    s = s.lower()
    # normalization 3: "&gt", "&lt"
    s = re.sub(r'&gt|&lt', ' ', s)
    # normalization 4: letter repetition (if more than 2)
    s = re.sub(r'([a-z])\1{2,}', r'\1', s)
    # normalization 5: non-word repetition (if more than 1)
    s = re.sub(r'([\W+])\1{1,}', r'\1', s)
    # normalization 6: string * as delimiter
    s = re.sub(r'\*|\W\*|\*\W', '. ', s)
    # normalization 7: stuff in parenthesis, assumed to be less informal
    s = re.sub(r'\(.*?\)', '. ', s)
    # normalization 8: xxx[?!]. -- > xxx.
    s = re.sub(r'\W+?\.', '.', s)
    # normalization 9: [.?!] --> [.?!] xxx
    s = re.sub(r'(\.|\?|!)(\w)', r'\1 \2', s)
    # normalization 10: ' ing ', noise text
    s = re.sub(r' ing ', ' ', s)
    # normalization 11: noise text
    s = re.sub(r'product received for free[.| ]', ' ', s)
    # normalization 12: phrase repetition
    s = re.sub(r'(.{2,}?)\1{1,}', r'\1', s)
    # remove all special characters
    s = re.sub('[^A-Za-z0-9]+', ' ', s)

    return s.strip()


## word level preprocess


# filtering out punctuations and numbers
def f_punct(w_list):
    """
    :param w_list: word list to be processed
    :return: w_list with punct and number filter out
    """
    return [word for word in w_list if word.isalpha()]


# selecting nouns
def f_noun(w_list):
    """
    :param w_list: word list to be processed
    :return: w_list with only nouns selected
    """
    return [word for (word, pos) in nltk.pos_tag(w_list) if pos[:2] == 'NN']

def f_stem(w_list):
    """
    :param w_list: word list to be processed
    :return: w_list with stemming
    """
    # stemming if doing word-wise
    p_stemmer = PorterStemmer()
    return [p_stemmer.stem(word) for word in w_list]



def f_stopw(w_list):
    """
    filtering out stop words
    """
    stop_words = (list(
        set(get_stop_words('en'))
    ))
    return [word for word in w_list if word not in stop_words]

# wrapper function around all word level proprocessing utilities
def preprocess_word(s):
    """
    Get word level preprocessed data from preprocessed sentences
    including: remove punctuation, select noun, stem, stop_words
    :param s: sentence to be processed
    :return: word level pre-processed list
    """
    w_list = word_tokenize(s)
    w_list = f_punct(w_list)
    w_list = f_noun(w_list)
    w_list = f_stem(w_list)
    w_list = f_stopw(w_list)

    return w_list


# wrapper function to preprocess all the sentences in a document
def preprocess_documents(file=None, verbose = True):
    """
    Read the file and preprocess the data

    file(str): path to the input file
    """

    if file is None:
        print ('Please provide a valid file path')
        return

    print('Reading the file')

    documents = pd.read_excel(file)

    # replace nans with empty spaces to avoid trouble in preprocessing
    documents = documents.replace(np.nan, '', regex=True)

    id = documents['User ID (anonymus)']
    docs = documents['Objective_Name'].astype(str) + ' \n ' + documents['Objective_Description'].astype(str) + ' \n ' + \
                  documents['Objective_Metric'].astype(str)

    #docs = documents['Objective_Name'].astype(str) + ' \n ' + documents['Objective_Metric'].astype(str)

    n_docs = len(docs)

    print('Preprocessing raw texts \n' )

    sentences_list = []  # sentence level preprocessed
    token_lists = []  # word level preprocessed

    id_drop = [] #place holder to save indexes for entries with non-english langauges to be dropped later.
    keep_track_of_id = []
    for idx in range(n_docs):
        text = docs[idx]

        try:
            detected_language = detect(text)

        except Exception as e:
            #print (e.msg)
            print ("Couldn't detect the language of the text on index {}".format(idx))
            print (text)
            # donot include undetected languages and drop their ids
            id_drop.append(idx)
            continue

        if  detected_language == 'en':

            sentences = nltk.sent_tokenize(text)
            for sentence in sentences:
                sentence = preprocess_sent(sentence)
                token_list = preprocess_word(sentence)
                keep_track_of_id.append(id[idx])
                if verbose:
                    print ('\n')
                    print (sentence)
                    print ("Candidate Tokens: ", token_list)

                sentences_list.append(sentence)
                token_lists.append(token_list)
        else:
            id_drop.append(idx)

    #drop the employee id of the non-english text
    if verbose:
        print("Objectives of following employees at the following indices are being dropped: ", id_drop)


    id = id.drop([id.index[i] for i in id_drop])
    print("Dropped {} entries with non english text".format(len(id_drop)))

    print('\n Preprocessing raw texts. Done!')
    
    #saving the preprocessing data in a file
    temp_df = pd.DataFrame()
    temp_df['id'] = keep_track_of_id
    temp_df['sentences'] = sentences_list
    temp_df['token_list'] = token_lists
    dr = 'output/preprocessed_data/'
    import os
    if not os.path.exists(dr):
        os.makedirs(dr)
    temp_df.to_pickle(dr + "use_this_data.pkl")
    return sentences_list, token_lists

#preprocess_documents("../data/PMP_Skills_Matching_Pilot_2018_2019_small.xlsx")