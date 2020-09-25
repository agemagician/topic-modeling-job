# Importing the required python modules
from models.embeddings_model import get_embeddings
from models.autoencoder_model import Autoencoder
from utilities.utils import generate_n_gram, clean_topics_from_clusters
from utilities.visualizations import get_employee_clusters, get_wordcloud,  create_hist
from datetime import datetime
import gensim
from gensim import corpora, models
from sklearn.cluster import KMeans
import numpy as np
import pandas as pd
from collections import Counter
from sklearn.manifold import TSNE
import torch
import pyLDAvis
from pyLDAvis import gensim as gs
import os


import nltk
nltk.download('wordnet')


from utilities.document_preprocessing import preprocess_documents
class Topic_Model:
    def __init__(self, base = 'LDA', k=3, use_AE = True, gamma = 1, embedding = None, saved_embedding_dir = None, verbose= False, embedding_batch_size=None):
        '''

        :param base (str): Probabilistic algorithm to be used. possible values: 'LDA'
        :param k (int): expected number of clusters in the data
        :param AE (bool): whether or not to use AutoEncoder
        :param embedding (str): which embedding to use. possible values 'bert', 'albert', 'roberta', 'xlnet', 'electra'
        :param saved_embedding_dir(str): path to the precomputed directory
        :param verbose (bool):
        :param embedding_batch_size (str): batch size to use when computed embeddings for a large data
        '''
        self.verbose = verbose
        self.k = k
        self.dictionary = None
        self.corpus = None
        self.cluster_model = None
        self.basemodel = None
        self.vec = {}
        self.gamma = gamma # parameter for reletive importance of lda
        self.embedding = None
        self.base = base

        if embedding is None and base is None:
            print ("Either enter a valid embedding or a valid base algorithm")
            return
        elif embedding is None:
            self.method = self.base
            self.id = self.method + '_k_' + str(self.k) + '_' + datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
        elif self.base is None:
            self.method =  "EMBEDDING"
            self.embedding = embedding
            self.id = self.method + '_k_' + str(self.k) + '_' + self.embedding + '_' + datetime.now().strftime(
                "%Y_%m_%d_%H_%M_%S")
        else:
            self.method = self.base + "_EMBEDDING"
            self.embedding = embedding
            self.id = self.method + '_k_' +  str(self.k) + '_' + self.embedding + '_' + datetime.now().strftime("%Y_%m_%d_%H_%M_%S")

        self.dir = 'output/{}/{}/'.format(self.method, self.id)
        #create the directory for saving outputs
        if not os.path.exists(self.dir):
            os.makedirs(self.dir)
        self.AE = None
        
        self.top_topics = None
        self.saved_embedding_dir = saved_embedding_dir
        self.embedding_batch_size = embedding_batch_size
        self.use_AE = use_AE

    def vectorize(self, sentences, token_lists, method=None):
        """
        Get vector representations from selected methods
        """


        if method == 'LSI' or method == 'LDA':

            print('Getting vector representations for {} '.format(self.base))

            # turn tokenized documents into a id <-> term dictionary
            self.dictionary = corpora.Dictionary(token_lists)
            if self.verbose:
                print ("Dictionary used for {} \n".format(self.base) )
                for key, value in  self.dictionary.items():
                    print (key, value)
            # convert tokenized documents into a document-term matrix
            self.corpus = [self.dictionary.doc2bow(text) for text in token_lists]
            if self.verbose:
                print ("\n Corpus used for {} \n".format(self.base) )
                for c in self.corpus:
                    print (c)

            if self.base == 'LSI':
                self.basemodel = gensim.models.lsimodel.LsiModel(self.corpus, num_topics=self.k, id2word=self.dictionary)
            elif self.base == 'LDA':
                self.basemodel = gensim.models.ldamodel.LdaModel(self.corpus, num_topics=self.k, id2word=self.dictionary,
                                                passes=20)

            def get_vec_base(model, corpus, k):
                """
                Get the Lsi vector representation (probabilistic topic assignments for all documents)
                :return: vec_lsi with dimension: (n_doc * n_topic)
                """
                n_doc = len(corpus)
                vec_base = np.zeros((n_doc, k))

                if self.base == 'LSI':
                    for i in range(n_doc):
                        for topic, lsi_score in self.basemodel[self.corpus[i]]:
                            vec_base[i, topic] = lsi_score

                elif self.base == 'LDA':

                    for i in range(n_doc):
                        # get the distribution for the i-th document in corpus
                        for topic, prob in model.get_document_topics(corpus[i]):
                            vec_base[i, topic] = prob

                return vec_base

            vec = get_vec_base(self.basemodel, self.corpus, self.k)
            if self.verbose:
                print ("\n \n Shape of {} Vector: {}".format(self.base, vec.shape) )
                print (vec)
                print ('\n According to {}, the top words for each topic should be, '.format(self.base))
                print (self.get_top_words_base(6))
            print('Getting vector representations for {}. Done! \n'.format(self.base))
            self.vec[self.base] = vec
            return vec


        elif method == 'EMBEDDING':

            print('\n Getting vector representations for {} '.format(self.embedding))

            if self.saved_embedding_dir is None:
                sentence_embedding = get_embeddings(self.embedding, sentences=sentences, batch_size = self.embedding_batch_size)
            else:
                # use previously generated embedding
                sentence_embedding = get_embeddings(saved_embedding_dir=self.saved_embedding_dir)
            vec = np.array(sentence_embedding)

            self.vec['EMBEDDING'] = vec
            return vec

        elif '_EMBEDDING' in method:

            # Get probability of a text belonging to a class from LDA/LSI
            vec_base = self.vectorize(sentences, token_lists, method=self.base)

            # Get sentence level embedding from Bert
            vec_embedding = self.vectorize(sentences, token_lists, method='EMBEDDING')

            # Concatenat them
            vec_base_embedding = np.c_[vec_base * self.gamma, vec_embedding]
            self.vec[method] = vec_base_embedding

            if self.verbose:
                print ("\n Concatenating both {}(topic distribution per topic) and "
                       "{}(sentence embeddings) vectors gives a new vector of shape {}"
                       " \n".format(self.base, self.embedding, vec_base_embedding.shape))

            return vec_base_embedding

    def fit_AE(self, vec, epoch):
        # If we want to use an autoencoder to learn latent representation
        if self.use_AE:

            # Learn the latent representation of data
            input_size = vec.shape[1]
            self.AE = Autoencoder(latent_dim=32, input_dim=input_size, epochs=epoch, batch_size=16)

            if self.verbose:
                print("Using the following autoencoder to learn the latent representation of the vector")

            print('Fitting Autoencoder')

            self.AE.fit(vec)
            print('Fitting Autoencoder Done!')

        vec = self.AE.model.predict(vec)
        if self.verbose:
            print('Latent Representation size: ', np.array(vec).shape)
        if torch.cuda.is_available():
            return vec.cpu().numpy()
        else:
            return vec.numpy()

    def fit(self, sentences, token_lists, epoch = 5):
        """
        Fit the topic model for selected method given the preprocessed data

        :return:
        """

        # use kmeans clustering to cluster the latent representations
        m_clustering = KMeans
        self.cluster_model = m_clustering(self.k)

        self.vectorize(sentences, token_lists, self.method)
        if self.method == "LDA" or self.method == "LSI":
            print (" {} Done.".format(self.base))
            return # no AE used

        if self.use_AE: # used only if there is an embedding used
            self.vec['ENCODER'] = self.fit_AE(self.vec[self.method], epoch=epoch)
            print('Clustering the latent representations')
            self.cluster_model.fit(self.vec['ENCODER'])
        else: # cluster the embedding representation
            self.cluster_model.fit(self.vec['EMBEDDING'])

        print('Clustering. Done!')

    def get_top_words_base(self, top=5):

        if not self.basemodel:
            print("Base model has not been trained yet")
            return None

        self.top_topics = dict()

        if self.base == 'LDA':
            for k in range(self.k):
                topics = self.basemodel.show_topic(topicid=k, topn=top)
                self.top_topics[k] = topics
        elif self.base == 'LSI':
            for k in range(self.k):
                topics = self.basemodel.show_topic(topicno=k, topn=top)
                self.top_topics[k] = topics

        return self.top_topics

    def compress_features(self, ids):
        '''
        Preprocessing (Dimensionality Reduction) for visualization

            Perform Dimensionality reduction on the learned features using the following steps:

            1) Merge all the vectors of each employee into one by taking average

            2) Assign the class label to each employee based on argmax on label counts.

            3) Further compress the average feature vector into 2d/3d space for visualization.

        :return: pandas dataframe with features vectors(averaged over all objectives for each employee),
        compress features vectors 2d. compressed feature vector 3d,
        assigned cluster label
        '''

        # Getting feature vectors and cluster labels assigned by the model
        if self.method == 'LDA':
            feature_vector = self.vec['LDA']
            # for each sentence, find out the cluster it belongs to
            cluster_label_per_document = np.argmax(np.array(feature_vector), axis=1)
        else:
            cluster_label_per_document = np.array(self.cluster_model.labels_)
            feature_vector = self.vec['ENCODER']
            if torch.cuda.is_available():
                feature_vector = feature_vector.cpu().numpy()
            else:
                feature_vector = feature_vector.numpy()

        unique_ids = list(set(ids))
        num_employee = len(unique_ids)

        # Create place holder for the analysis
        averaged_employee_features = pd.DataFrame(None, index=pd.Index(unique_ids, name='Employee id'),
                                              columns=pd.Index(
                                                  ['feature_vector', 'cluster', 'feature_vector_2d',
                                                   'feature_vector_3d'],
                                                  name='Compressed_features'))
        # get the indices of all objectives/documents/sentence for each employee (for later use)
        employee_objective_indices = dict()
        for employee in unique_ids:
            employee_indices = [i for i, e in enumerate(ids) if (ids[i] == employee)]
            employee_objective_indices[employee] = employee_indices

        # 1 & 2 take average of all feature vectors belonging to an employee and assigning cluster label by taking the mode

        for employee in unique_ids:
            # get all feature vectors belonging to an employee
            feature_vector_per_employee = feature_vector[employee_objective_indices[employee]]
            # get label for all feature vectors belonging to that employee
            clusters_per_employee = cluster_label_per_document[employee_objective_indices[employee]]

            # take mean of all feature vectors
            averaged_employee_features['feature_vector'][employee] = np.mean(feature_vector_per_employee, axis=0)

            # pick the label with most occurance for that employee
            c = Counter(clusters_per_employee)
            averaged_employee_features['cluster'][employee] = c.most_common(1)[0][0]

        # 3) doing dimensionality reduction into 2d/3d Space
        # todo: find a more permanent solution for this
        # panda returns a array of array for the column feature vector. Converting from array of array to 2d array
        feature_vector_ = np.stack(averaged_employee_features['feature_vector'].to_numpy())

        # reduce dimension of feature vectors into 2
        feature_vector_2d = TSNE(n_components=2).fit_transform(feature_vector_)
        averaged_employee_features['feature_vector_2d'] = list(feature_vector_2d)

        # reduce dimension of feature vectors into 3
        feature_vector_3d = TSNE(n_components=3).fit_transform(feature_vector_)
        averaged_employee_features['feature_vector_3d'] = list(feature_vector_3d)

        return averaged_employee_features

    def wordcloud(self, sentences, algorithm):
        '''
        Visualization Function
        This functions creates wordclouds as per the specified configurations through the user interface
        and saves them in the predefined directory
        :param TopicModelObject: TopicModel instances
        :return:
        '''

        if self.method == 'LDA':
            # we use the word_distribution method for word cloud as we only have probability
            # of that word belonging to the cluster from LDA
            for k, cluster_word_distribution in self.get_top_words_lda(top=100).items():
                get_wordcloud(word_distribution=cluster_word_distribution, topic=k, dir=self.dir)
        else:
            # for LDA + Bert, we have no distribution so we pick the most occuring words from the
            # sentences belonging to that cluster

            # Getting indexes of sentences belonging to each cluster
            sentence_label_dict = dict()
            for label in set(self.cluster_model.labels_):
                sentence_label_dict[label] = [idx for idx, l in enumerate(self.cluster_model.labels_) if
                                              l == label]
            for i in range(self.k):
                k_sentences = np.array(sentences)[sentence_label_dict[i]]
                ngram_token_with_frequency = generate_n_gram(k_sentences, n=algorithm['n_gram'], separator='-',
                                                             keep_stopwords=False, stem=False, keep_nouns_only=False)

                get_wordcloud(topic=i, word_count_dict=ngram_token_with_frequency, dir=self.dir)

    def get_labels(self):
        if self.method == 'LDA' or self.method == 'LSI':
            return np.argmax(self.vec[self.base], axis=1)
        else:
            return self.cluster_model.labels_
    def get_cluster_text(self, sentences, ngram):

        # Getting indexes of sentences belonging to each cluster
        sentence_label_dict = dict()

        labels = self.get_labels()
        for label in set(labels):
            sentence_label_dict[label] = [idx for idx, l in enumerate(labels) if l == label]
        # constructing cluster titles
        cluster_titles = []
        for i in range(self.k):
            k_sentences = np.array(sentences)[sentence_label_dict[i]]
            ngram_token_with_frequency = Counter(generate_n_gram(k_sentences, n=ngram, separator='-',
                                                                 keep_stopwords=True, stem=False,
                                                                 keep_nouns_only=False))

            cluster_title = ', '.join([key for key, value in ngram_token_with_frequency.most_common(6)])

            cluster_titles.append(cluster_title)
        return cluster_titles

    def get_employee_text(self,sentences, ids):
        # get the indices of all objectives/documents/sentence for each employee
        employee_objective_indices = dict()
        for employee in set(ids):
            employee_indices = [i for i, e in enumerate(ids) if (ids[i] == employee)]
            employee_objective_indices[employee] = employee_indices

        np_sentences = np.array(sentences)
        employee_titles = dict()
        for emp in list(set(ids)):
            emp_indices = employee_objective_indices[emp]
            employee_title = '. '.join([objective for objective in np_sentences[emp_indices]])

            employee_titles[emp] = emp + ': ' + employee_title[0:150]
        return employee_titles

    def vis_employee_clusters(self, sentences, ids, vis_type, ngram):
        '''
        Visualization Function
        :param sentences:
        :param ids:
        :param vis_type:
        :param ngram:
        :return:
        '''
        averaged_employee_features = self.compress_features(ids)

        cluster_titles = self.get_cluster_text(sentences,ngram)

        # constructing hover text (all objectives belonging to that employee)
        employee_titles = self.get_employee_text(sentences,ids)

        #calling the visualization function
        get_employee_clusters(averaged_employee_features, cluster_titles, employee_titles, vis=vis_type, dir = self.dir)

    def vis_pyldavis(self, token_lists):
        '''

        :param token_lists:
        :return:
        '''

        id2word = corpora.Dictionary(token_lists)
        corpus = [id2word.doc2bow(text) for text in token_lists]

        # The visualization call is straight forward so no need to create a separate function for it in visualization.py
        vis = gs.prepare(self.ldamodel, corpus, id2word)
        pyLDAvis.show(vis)

    def quick_stats(self, ids):
        '''
        returns the distribution of employees in each cluster
        :param ids:
        :param cluster_label_per_document:
        :return:
        '''

        unique_ids = list(set(ids))

        #get cluster prediction for all employees
        if self.method == 'LDA':
            feature_vector = self.vec[self.method]
            # for each sentence, find out the cluster it belongs to
            cluster_label_per_document = np.argmax(np.array(feature_vector), axis=1)
        else:
            cluster_label_per_document = np.array(self.cluster_model.labels_)
        # create a dataframe to store the frequence per cluster for each employee.
        df_id_cluster = pd.DataFrame(0,
                                     index=pd.Index(unique_ids, name='Employee id'),
                                     columns=pd.Index([i for i in range(self.k)], name='Clusters'))

        # There are multiple entries per employee so an employee can belong to multiple clusters
        for i, _id in enumerate(ids):
            df_id_cluster[cluster_label_per_document[i]][_id] = 1

        #shows how many employees belong to each cluster
        employee_cluster_frequency = np.sum(df_id_cluster.to_numpy(), axis=0)

        # shows percentage of employees belong to each cluster
        employee_cluster_frequency = (employee_cluster_frequency/len(unique_ids))*100
        print (employee_cluster_frequency)
        create_hist(employee_cluster_frequency)

#file = "/home/anum/Downloads/25828_32915_compressed_Train_rev1.csv/Train_rev1.csv"


#ids, sentences, token_lists = preprocess_documents(file, verbose = False,
#                                                   dr = "../output/preprocessed_data/job/",
#                                                   subset = 5000)


data_file = "../output/preprocessed_data/job/use_this_data_5000.pkl"

import pickle5 as pickle

with (open(data_file, "rb")) as openfile:
    data = pickle.load(openfile)

sentences = list(data['sentences'])
token_lists = list(data['token_list'])
ids = list(data['id'])

#sentences = sentences[0:200]
#ids = ids[0:200]
#token_lists = token_lists[0:200]


num_docs = len(ids)

# Define the topic model object
TopicModelObject = Topic_Model(base = None, k = 3, verbose = False, embedding = 'bert',
                               saved_embedding_dir = 'output/embeddings/bert_job_5000/bert_embeddings.json',
                               embedding_batch_size = 500, use_AE= False)
TopicModelObject.fit(sentences, token_lists, epoch=5)

# Getting indexes of sentences belonging to each cluster
sentence_label_dict = dict()

if TopicModelObject.method == 'LDA':
    labels = np.argmax(TopicModelObject.vec['LDA'], axis=1)
else:
    labels = TopicModelObject.cluster_model.labels_

for label in set(labels):
    sentence_label_dict[label] = [idx for idx, l in enumerate(labels) if l == label]



### Top words from LDA
if TopicModelObject.method == 'LDA':

    lda_class_distribution = TopicModelObject.vec['LDA']
    labels = np.argmax(lda_class_distribution, axis=1)

    for k, cluster_word_distribution in TopicModelObject.get_top_words_lda(top=100).items():

        cluster_word_distribution_dict = dict()
        for item in cluster_word_distribution:
            cluster_word_distribution_dict[item[0]] = item[1]
        get_wordcloud(dir=TopicModelObject.dir, word_distribution=cluster_word_distribution_dict, topic=k)
# Get top words from autoencoder (LDA + Embeddings) and create word clouds on n grams
else:

    grams = 1
    for ng in range(grams):
        topic_words = []
        print("WordClouds for {} grams".format(str(ng + 1)))
        # get topic words for all clusters
        for i in range(TopicModelObject.k):
            k_sentences = np.array(sentences)[sentence_label_dict[i]]
            ngram_token_with_frequency = generate_n_gram(k_sentences, n=ng + 1, separator='-',
                                                         keep_stopwords=False, stem=True, keep_nouns_only=True,
                                                         top=100)
            topic_words.append(ngram_token_with_frequency)

        # remove simliar words from clusters
        topic_words = clean_topics_from_clusters(topics=topic_words)

        # generate wordclouds
        for i in range(TopicModelObject.k):
            get_wordcloud(topic=i, word_count_dict=topic_words[i], dir=TopicModelObject.dir)

print ('halt')