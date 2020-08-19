from flask import render_template, flash, request, redirect, url_for
from flaskApp import app
import pandas as pd

import os
import shutil

#imports from my own classes
from utilities.document_preprocessing import preprocess_documents

from models.topic_model import Topic_Model

from models.embeddings_model import get_embeddings

@app.route('/', methods=['GET'])
@app.route('/home')
def home():

    return render_template('home.html', user=user, embeddings=embeddings, base_models = base_models,
                           num_clusters=num_clusters, n_grams = n_grams, visualizations=visualizations)

@app.route('/upload', methods=['GET', 'POST'])
def upload():
    return render_template('upload.html')

@app.route('/precompute', methods=['GET', 'POST'])
def precompute():
    if request.method == 'POST':
        success = False
        #uploading the file
        try:
            upload = request.files.get("file")
            print("file name: {}".format(upload.filename))
            filename = upload.filename
            destination = temp_data_folder + filename
            upload.save(destination)
            print("Uploaded to ", destination)

            #reading and preprocessing the data
            ids, sentences, token_lists = preprocess_documents(destination, verbose=False, dr = temp_data_folder)
            #generating embeddings and saving it
            for embed in embeddings:
                print (embed)
                if embed == 'none':
                    continue
                sentence_embedding = get_embeddings(model_name=embed, sentences=sentences,
                                                    batch_size=100, dr = temp_data_folder)

            success = True

        except:
            return render_template('precompute.html', filename="Failed to precompute data")

    #delete the previous data
    if success:
        if os.path.exists(data_folder):
            os.remove(data_folder)
        os.rename(temp_data_folder, data_folder)
        os.remove(temp_data_folder)

    return render_template('precompute.html', filename=filename)
@app.route("/result" , methods=['GET', 'POST'])
def result():

    #define few global variables to be used throughout
    global sentences, algorithm

    #reading the configurations
    algorithm = dict()

    algorithm['base'] = str(request.form.get('selected_base_model'))
    algorithm['k'] = int(request.form.get('selected_k'))
    algorithm['n_gram'] =int( request.form.get('selected_ngram'))
    algorithm['visualization'] = str(request.form.get('selected_vis'))

    if request.form.get('selected_embedding') == 'none':
        algorithm['embedding'] = None
        embedding_path= None
    else:
        algorithm ['embedding'] = str(request.form.get('selected_embedding'))
        # computing path to the selected embedding
        embedding_path = data_folder + algorithm['embedding'] + "_embeddings.json"

    # reading the data
    data = pd.read_pickle(data_folder + processed_file)
    sentences = list(data['sentences'])
    token_lists = list(data['token_list'])
    ids = list(data['id'])

    # Define the topic model object
    TopicModelObject = Topic_Model(k=algorithm['k'], verbose=False, embedding=algorithm['embedding'],
                                   saved_embedding_dir=embedding_path)
    # Fit the topic model
    TopicModelObject.fit(sentences, token_lists)

    if algorithm['visualization'] == 'Word Cloud':
        # generating and saving the word clouds for the defined clusters
        TopicModelObject.wordcloud(sentences,algorithm)
    elif "Employee Cluster" in algorithm['visualization']:
        # if visualization is Employee Clusters using plotly
        TopicModelObject.vis_employee_clusters(sentences, ids, algorithm['visualization'], algorithm['n_gram'])
    else:
        # Network visuaization
        TopicModelObject.vis_network(ids,sentences, algorithm['n_gram'])

    # get list of all word cloud images from the directory
    images = os.listdir(TopicModelObject.dir)

    # moving them to the static folder to be accessed by Flask
    # getting source paths
    source = TopicModelObject.dir
    #getting destination path
    destination = 'flaskApp/static/' + TopicModelObject.dir
    if not os.path.exists(destination):
        os.makedirs(destination)
    # move the files
    for f in images:
        shutil.move(source + f, destination)
    #constructing path to the images inside the static folder
    images = [{'id': i+1, 'path': TopicModelObject.dir + '/' + image}
                        for i , image in enumerate(images)]

    return render_template('results.html', algorithm=algorithm, images = images)



user = {'username': 'Anum'}
embeddings = ['bert', 'xlnet', 'none']
base_models = ['Latent Drichlet Allocation - LDA']
n_grams = [1, 2, 3, 4]
num_clusters = [1,2,3,4,5,6,7,8,9,10]
visualizations = ["Word Cloud", " Networks", "Employee Cluster 2D", "Employee Cluster 3D"]

#define the directory where data is stored temporarily
temp_data_folder = "data/temp/"
#define the permanent folder for the data
data_folder = "data/flask/"
processed_file = "use_this_data.pkl"
if not os.path.exists(temp_data_folder):
    os.makedirs(temp_data_folder)
