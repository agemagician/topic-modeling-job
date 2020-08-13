import umap
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import numpy as np
import os
import plotly.graph_objects as go

#todo: move this function inside the topic model class
def get_wordcloud(model, token_lists=None, topic=None, word_distribution=None, word_count_dict=None):
    """
    Get word cloud of each topic from fitted model
    :param model: Topic_Model object
    :param sentences: preprocessed sentences from docs
    """

    print('Getting wordcloud for topic {} ...'.format(topic))

    if word_count_dict is not None:
        wordcloud = WordCloud(width=800, height=560,
                              background_color='white', collocations=False,
                              min_font_size=10).generate_from_frequencies(word_count_dict)


    elif word_distribution is not None:
        tokens = str()
        for item in word_distribution:
            word = item[0]
            dist = item[1]
            # from porobability to actually numbers the word appears in the document
            word_occurance = int (dist * 1000 )
            tokens += ' '.join([''.join(word) for _ in range(word_occurance)])
            tokens += ' '

        wordcloud = WordCloud(width=800, height=560,
                              background_color='white', collocations=False,
                              min_font_size=10).generate(tokens)

    else:
        lbs = model.cluster_model.labels_
        tokens = ' '.join([' '.join(_) for _ in np.array(token_lists)[lbs == topic]])

        wordcloud = WordCloud(width=800, height=560,
                          background_color='white', collocations=False,
                          min_font_size=10).generate(tokens)

    # plot the WordCloud image
    plt.figure(figsize=(8, 5.6), facecolor=None)
    plt.imshow(wordcloud)
    plt.axis("off")
    plt.tight_layout(pad=0)
    if not os.path.exists(model.dir):
        os.makedirs(model.dir)
    plt.savefig(model.dir + 'Topic' + str(topic) + '_wordcloud')
    print('Getting wordcloud for topic {}. Done!'.format(topic))

#todo: move this function inside the topic model class
def get_employee_clusters(model, averaged_employee_features, cluster_titles , vis = 'Employee Cluster 2D'):
    if not os.path.exists(model.dir):
        os.makedirs(model.dir)

    fig = go.Figure()

    if vis == 'Employee Cluster 2D':
        for i, cl in enumerate(range(model.k)):
            cluster_df = averaged_employee_features[averaged_employee_features.cluster == cl]

            # plot all the employees belonging to a cluster using a new trace.
            features = np.stack(cluster_df['feature_vector_2d'].to_numpy())


            fig.add_trace(go.Scattergl(
                x=features[:, 0],
                y=features[:, 1],
                mode='markers',
                showlegend=True,
                text=cluster_df.index.values,
                marker=dict(size=8, color=i),
                name="cluster_" + str(cl + 1) + ": " + cluster_titles[cl]
            ))

        fig.update_layout(title='Distribution of employees over clusters in 2d space', legend=dict(
            x=0.6,
            y=1.4,
            traceorder='normal',
            font=dict(
                size=10)))
        path = model.dir + '2D_clusters.jpeg'
    else:
        #vis is 3d (Employee Cluster 3D)
        fig = go.Figure()

        for i, cl in enumerate(range(model.k)):
            cluster_df = averaged_employee_features[averaged_employee_features.cluster == cl]

            # plot all the employees belonging to a cluster using a new trace.
            features = np.stack(cluster_df['feature_vector_3d'].to_numpy())

            fig.add_trace(go.Scatter3d(
                x=features[:, 0],
                y=features[:, 1],
                z=features[:, 2],
                mode='markers',
                showlegend=True,
                text=cluster_df.index.values,
                marker=dict(size=3, color=i),
                name="cluster_" + str(cl + 1) + ": " + cluster_titles[cl]
            ))

        fig.update_layout(title='Distribution of employees over clusters in 3d Space', legend=dict(
            x=0.6,
            y=1.4,
            traceorder='normal',
            font=dict(
                size=10)))
        path = model.dir + '3D_clusters.jpeg'

    fig.show()
    fig.write_image(path)
    print ("Saved Clusters to " , path)

