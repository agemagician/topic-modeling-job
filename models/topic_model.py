# Importing the required python modules
from models.embeddings_model import get_embeddings
from models.autoencoder_model import Autoencoder


from datetime import datetime
import gensim
from gensim import corpora, models
from sklearn.cluster import KMeans
import numpy as np

class Topic_Model:
    def __init__(self, k=3, embedding = None, saved_embedding_dir = None, verbose= False, embedding_batch_size=100):
        """
        :param k: number of topics
        """
        self.verbose = verbose
        self.k = k
        self.dictionary = None
        self.corpus = None
        self.cluster_model = None
        self.ldamodel = None
        self.vec = {}
        self.gamma = 15  # parameter for reletive importance of lda
        self.embedding = None

        if embedding is None:
            self.method = "LDA"
            self.id = self.method + '_k_' + str(self.k) + '_' + datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
        else:
            self.method = "LDA_EMBEDDING"
            self.embedding = embedding
            self.id = self.method + '_k_' +  str(self.k) + '_' + self.embedding + '_' + datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
        
        self.dir = 'output/{}/{}/'.format(self.method, self.id)
        self.AE = None
        
        self.top_topics = None
        self.saved_embedding_dir = saved_embedding_dir
        self.embedding_batch_size = embedding_batch_size

    def vectorize(self, sentences, token_lists, method=None):
        """
        Get vector representations from selected methods
        """

        if method == 'LDA':

            print('Getting vector representations for LDA ')

            # turn tokenized documents into a id <-> term dictionary
            self.dictionary = corpora.Dictionary(token_lists)
            if self.verbose:
                print ("Dictionary used for LDA \n" )
                for key, value in  self.dictionary.items():
                    print (key, value)
            # convert tokenized documents into a document-term matrix
            self.corpus = [self.dictionary.doc2bow(text) for text in token_lists]
            if self.verbose:
                print ("\n Corpus used for LDA \n" )
                for c in self.corpus:
                    print (c)

            self.ldamodel = gensim.models.ldamodel.LdaModel(self.corpus, num_topics=self.k, id2word=self.dictionary,
                                                                passes=20)

            def get_vec_lda(model, corpus, k):
                """
                Get the LDA vector representation (probabilistic topic assignments for all documents)
                :return: vec_lda with dimension: (n_doc * n_topic)
                """
                n_doc = len(corpus)
                vec_lda = np.zeros((n_doc, k))
                for i in range(n_doc):
                    # get the distribution for the i-th document in corpus
                    for topic, prob in model.get_document_topics(corpus[i]):
                        vec_lda[i, topic] = prob

                return vec_lda

            vec = get_vec_lda(self.ldamodel, self.corpus, self.k)
            if self.verbose:
                print ("\n \n Shape of LDA Vector: ", vec.shape)
                print (vec)
                print ('\n According to LDA, the top words for each topic should be, ')
                print (self.get_top_words_lda(6))
            print('Getting vector representations for LDA. Done! \n')
            self.vec["LDA"] = vec
            return vec

        elif method == 'EMBEDDING':

            print('\n Getting vector representations for {} '.format(self.embedding))

            if self.saved_embedding_dir is None:
                sentence_embedding = get_embeddings(self.embedding, sentences=sentences, batch_size = self.embedding_batch_size)
            else:
                # use previously generated embedding
                sentence_embedding = get_embeddings(saved_embedding_dir=self.saved_embedding_dir)
            vec = np.array(sentence_embedding)

            return vec

        elif method == 'LDA_EMBEDDING':

            # Get probability of a text belonging to a class from LDA
            vec_lda = self.vectorize(sentences, token_lists, method='LDA')

            # Get sentence level embedding from Bert
            vec_embedding = self.vectorize(sentences, token_lists, method='EMBEDDING')
            print (vec_lda.shape)
            print (vec_embedding.shape)
            # Concatenat them
            vec_lda_embedding = np.c_[vec_lda * self.gamma, vec_embedding]
            self.vec['LDA_EMBEDDING'] = vec_lda_embedding

            if self.verbose:
                print ("\n Concatenating both LDA(topic distribution per topic) and "
                       "{}(sentence embeddings) vectors gives a new vector of shape {}"
                       " \n".format(self.embedding, vec_lda_embedding.shape))

            if not self.AE:

                # Learn the latent representation of data
                input_size = vec_lda.shape[1] + vec_embedding.shape[1]
                self.AE = Autoencoder(latent_dim=32, input_dim=input_size, epochs=5)

                if self.verbose:
                    print ("Using the following autoencoder to learn the latent representation of the vector")

                print('Fitting Autoencoder')
                self.AE.fit(vec_lda_embedding)
                print('Fitting Autoencoder Done!')

            vec = self.AE.model.predict(vec_lda_embedding)
            if self.verbose:
                print ('Latent Representation size: ', np.array(vec).shape)
            return vec

    def fit(self, sentences, token_lists):
        """
        Fit the topic model for selected method given the preprocessed data

        :return:
        """

        # use kmeans clustering to cluster the latent representations
        m_clustering = KMeans
        self.cluster_model = m_clustering(self.k)

        self.vec[self.method] = self.vectorize(sentences, token_lists, self.method)
        if self.method == "LDA":
            print (" LDA Done.")
            return
        print('Clustering the latent representations')
        
        self.cluster_model.fit(self.vec[self.method].cpu())

        print('Clustering. Done!')

    def get_top_words_lda(self, top=5):

        if not self.ldamodel:
            print("LDA model has not been trained yet")
            return None

        self.top_topics = dict()

        for k in range(self.k):
            topics = self.ldamodel.show_topic(topicid=k, topn=top)
            self.top_topics[k] = topics
        return self.top_topics