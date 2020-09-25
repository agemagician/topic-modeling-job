import nltk
from stop_words import get_stop_words
from nltk.stem.porter import PorterStemmer
from nltk.stem.wordnet import WordNetLemmatizer
from utilities.visualizations import create_hist
from collections import Counter
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd

def clean_topics_from_clusters(topics):
    K = len(topics)

    topics_to_delete = []
    for _ in range(K):
        topics_to_delete.append([])
    # interate over the topics of all clusters
    for k, topic_k in enumerate(topics):
        for word_k, count_k in topic_k.items():
            #for each word in the topic, check if it exists in the other clusters
            for ind_j, topic_j in enumerate(topics[k+1:K]):
                if word_k in topic_j.keys(): # if the word exists in other clusters too

                    if topic_k[word_k] > topic_j[word_k]: #the word has more frequency in initial cluster
                        topics_to_delete[ind_j+k+1].append(word_k)
                    elif topic_k[word_k] < topic_j[word_k]: # topic has less frequency in initial cluster
                        topics_to_delete[k].append(word_k) # so we delete it from initial cluster


    topics_clean = []
    for i , topic in enumerate(topics):
        for word in topics_to_delete[i]:
            if word in topic.keys():
                del topic[word]
        topics_clean.append(topic)


    ## use tf-idf to discard non-unqiue / common words from cluster
    '''
    tfidf_sentences = []
    for topic in topics:
        clean_sentence = ' '.join(word for word in topic)
        tfidf_sentences.append(clean_sentence)

    tfidf = TfidfVectorizer()
    tfidf_vec = tfidf.fit_transform(tfidf_sentences)
    tfidf_features = tfidf.get_feature_names()

    pd_tfidf_vec = pd.DataFrame.sparse.from_spmatrix(tfidf_vec)
    tfidf_topics = []
    for k in range(K):
        k_topics = np.array(pd_tfidf_vec.iloc[[k]])
        top_k_topic_ind = np.argsort(k_topics)
        top_k_topic_ind = top_k_topic_ind[:,-50:]

        #using the ids above, get feature names and append them the list
        topic_k = []
        for top in top_k_topic_ind[0]:
            topic_k.append(tfidf_features[int(top)])
        tfidf_topics.append(topic_k)



    tfidf_topics_clean = []
    for i , topic in enumerate(tfidf_topics):
        tfidf_topic = dict()
        for word in topic:
            if word in topics[i].keys():
                tfidf_topic[word] = topics[i][word]
        tfidf_topics_clean.append(tfidf_topic)
    '''
    return topics_clean
    #return tfidf_topics_clean
    #return tfidf_topics



def generate_n_gram(sentences, n=2,separator = '-',  keep_stopwords = True, stem=False, keep_nouns_only = False, top = 30):
    '''
    personal thoughts: A very brain consuming function I had written in a while.

    Given a list of sentences, it generates a list of all n-grams
    :param sentences: a list of sentences to generate n-grams of
    :param n: (int) gram value
    :param top (int): top number of words to be used
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

        tokenized_sent = nltk.word_tokenize(sent)

        if keep_stopwords == False:
            tokenized_sent = [word for word in tokenized_sent if word not in stop_words]

        if stem == True:
            wnl = WordNetLemmatizer()
            tokenized_sent = [wnl.lemmatize(word) for word in tokenized_sent]

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

                n_grams_all_sentences.append(feature)


    ngram_token_with_frequency = Counter(n_grams_all_sentences).most_common()
    return dict(ngram_token_with_frequency)

def length_analysis(objectives):
    '''
    given a list of all objective(text -> str), it returns a list with their respective lengths and also plots it ina histogram
    :param objectives: list of all objective
    :return: length (list) list of integers representing the number of words in each objective
    '''
    objectives = list(objectives)
    length = [len(obj.split()) for obj in objectives]
    length.sort()

    print ("Shortest Objective is of length ", min(length))
    print("Largest Objective is of length ", max(length))

    # how many objectives have a length of x
    count_frequency_dict = {x: length.count(x) for x in length}

    # find the maximum allowed length of the objective (in terms of word cound)
    # by finding 'allowed_len' which is the upper cap of the 98% of the objectives lens.
    #since we have to pad the sentences when getting embeddings, some outliers have a very long length which is noise for
    # all the remaining 98% of the objectives
    allowed_percentage = 0.98
    total_length = len(objectives) # count of total objectives
    allowed_count = total_length*allowed_percentage
    allowed_length = 0 # objectives word count would be truncated at this length
    count = 0
    for key, value in count_frequency_dict.items():
        if count >= allowed_count: # if the count has reached the allowed percentage of total length
            allowed_length = key
            break
        count += value


    # use the function from utilities.visualizations
    create_hist(length,'Number of words per objective', 'Length of Objective', 'Number of Objectives' , save=True )




    return allowed_length

#t =[[1,55,3,6,9,2], [1,5,7,3,2,9], [23, 12,66,32,8,4,], [11,44,55,18,30,888]]
#clean_topics_from_clusters(t)