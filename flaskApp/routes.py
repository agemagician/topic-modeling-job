from flask import render_template, request
from flaskApp import app
import pandas as pd
from models.topic_model import Topic_Model
from utilities.visualizations import *
import os
import shutil


@app.route('/', methods=['GET'])
@app.route('/index')
def index():
    user = {'username': 'Anum'}
    embeddings = ['BERT', 'XLNET', 'None']
    base_models = ['Latent Drichlet Allocation - LDA']
    num_clusters = [1,2,3,4,5,6,7,8,9,10]

    return render_template('index.html', user=user, embeddings=embeddings, base_models = base_models, num_clusters=num_clusters)

@app.route("/result" , methods=['GET', 'POST'])
def result():
    algorithm = dict()

    algorithm['base'] = request.form.get('selected_base_model')
    algorithm['k'] = request.form.get('selected_k')

    if request.form.get('selected_embedding') == 'None':
        algorithm['embedding'] = None
        embedding_path = None
    else:
        algorithm ['embedding'] = str(request.form.get('selected_embedding')).lower()
        # computing path to the selected embedding
        embedding_path = "output/precomputed/" + algorithm['embedding'] + "_embeddings.json"

    # reading the data
    data = pd.read_pickle("output/precomputed/use_this_data.pkl")
    sentences = list(data['sentences'])
    token_lists = list(data['token_list'])

    # Define the topic model object
    TopicModelObject = Topic_Model(k=algorithm['k'], verbose=False, embedding=algorithm['embedding'],
                                   saved_embedding_dir=embedding_path)
    # Fit the topic model
    TopicModelObject.fit(sentences, token_lists)

    # generating and saving the word clouds for the defined clusters
    if TopicModelObject.method == 'LDA':
        # we use the word_distribution method for word cloud as we only have probability
        # of that word belonging to the cluster from LDA
        for k, cluster_word_distribution in TopicModelObject.get_top_words_lda(top=100).items():
            get_wordcloud(TopicModelObject, word_distribution=cluster_word_distribution, topic=k)
    else:
        # for LDA + Bert, we have no distribution so we pick the most occuring words from the
        # sentences belonging to that cluster
        for i in range(TopicModelObject.k):
            get_wordcloud(TopicModelObject, token_lists, i)

    # get list of all word cloud images from the directory
    wordcloud_images = os.listdir(TopicModelObject.dir)

    # moving them to the static folder to be accessed by Flask
    # getting source paths
    source = TopicModelObject.dir
    #getting destination path
    destination = 'flaskApp/static/' + TopicModelObject.dir
    if not os.path.exists(destination):
        os.makedirs(destination)
    # move the files
    for f in wordcloud_images:
        shutil.move(source + f, destination)
    #constructing path to the images inside the static folder
    wordcloud_images = [TopicModelObject.dir + '/' + image for image in wordcloud_images]
    print (wordcloud_images)

    return render_template('results.html', algorithm=algorithm, wordcloud_images = wordcloud_images)