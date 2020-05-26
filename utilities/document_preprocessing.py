import re
import nltk
from nltk.stem.porter import PorterStemmer
from stop_words import get_stop_words
from nltk.tokenize import word_tokenize
import pandas as pd
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
    documents = pd.read_csv(file)
    docs = documents.paragraph
    n_docs = len(docs)

    print('Preprocessing raw texts \n' )

    sentences = []  # sentence level preprocessed
    token_lists = []  # word level preprocessed

    for idx in range(n_docs):
        sentence = preprocess_sent(docs[idx])
        token_list = preprocess_word(sentence)
        if verbose:
            print ('\n')
            print (sentence)
            print ("Candidate Tokens: ", token_list)

        sentences.append(sentence)
        token_lists.append(token_list)

    print('\n Preprocessing raw texts. Done!')
    return sentences, token_lists