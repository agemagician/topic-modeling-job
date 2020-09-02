import nltk
from stop_words import get_stop_words
from nltk.stem.porter import PorterStemmer
from utilities.visualizations import create_hist

def generate_n_gram(sentences, n=2,separator = '-',  keep_stopwords = True, stem=False, keep_nouns_only = False):
    '''
    personal thoughts: A very brain consuming function I had written in a while.

    Given a list of sentences, it generates a list of all n-grams
    :param sentences: a list of sentences to generate n-grams of
    :param n: (int) gram value
    :return: n_gram_per_sentence: (list) a list of list (which corresponds to each sentence) and contains n-gram token
    features of the respective sentence
    '''

    stop_words = (list(
        set(get_stop_words('en'))
    ))

    # n for grams cannot be negative
    if n < 1:
        print ("n must be greater than or equal to 1")
        return None

    n_grams_all_sentences = []
    insufficient_count_for_feature = n-1

    for sent in sentences:
        n_gram_per_sentence = []
        tokenized_sent = nltk.word_tokenize(sent)

        if keep_stopwords == False:
            tokenized_sent = [word for word in tokenized_sent if word not in stop_words]

        if stem == True:
            p_stemmer = PorterStemmer()
            tokenized_sent = [p_stemmer.stem(word) for word in tokenized_sent]

        if keep_nouns_only == True:
            tokenized_sent = [word for (word, pos) in nltk.pos_tag(tokenized_sent) if pos[:2] == 'NN']

        tokenized_sent = [word for word in tokenized_sent if word.isalpha()]

        #only if there are enough tokens in the sentence to create at least one feature
        if len(tokenized_sent) >= n:
            for token_number in range(len(tokenized_sent)-insufficient_count_for_feature):

                feature = "" #empty place holder for building n-gram
                for i in range(n):
                    feature += tokenized_sent[token_number + i]
                    # append a "-" only if more tokens need to be added to the feature
                    if i < n-1:
                        feature += separator

                n_gram_per_sentence.append(feature)
        n_grams_all_sentences.append(n_gram_per_sentence)

    #todo: replace this with a counter maybe?
    ngram_token_with_frequency = dict()

    for sentence in n_grams_all_sentences:
        for feature in sentence:
            if feature in ngram_token_with_frequency.keys():
                ngram_token_with_frequency[feature] += 1
            else:
                ngram_token_with_frequency[feature] = 1

    return ngram_token_with_frequency

def length_analysis(objectives):
    '''
    given a list of all objective(text -> str), it returns a list with their respective lengths and also plots it ina histogram
    :param objectives: list of all objective
    :return: length (list) list of integers representing the number of words in each objective
    '''
    objectives = list(objectives)
    length = [len(obj.split()) for obj in objectives]

    print ("Shortest Objective is of length ", min(length))
    print("Largest Objective is of length ", max(length))

    # use the function from utilities.visualizations
    create_hist(length,'Number of words per objective', 'Length of Objective', 'Number of Objectives' , save=True )

    return length