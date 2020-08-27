
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import numpy as np
import os
import plotly.graph_objects as go

#todo: move this function inside the topic model class
def get_wordcloud(token_lists=None, topic=None, word_distribution=None, word_count_dict=None, dir = os.getcwd()):
    """
    Get word cloud of each topic from fitted model
    :param model: Topic_Model object
    :param sentences: preprocessed sentences from docs
    """

    print('Getting wordcloud for topic {} ...'.format(topic))

    # LDA + embeddings
    if word_count_dict is not None:
        wordcloud = WordCloud(width=800, height=560,
                              background_color='white', collocations=False,
                              min_font_size=10).generate_from_frequencies(word_count_dict)

    #LDA
    elif word_distribution is not None:
        tokens = str()
        for item in word_distribution:
            word = item[0]
            dist = item[1]
            # from probability to actually numbers the word appears in the document
            word_occurance = int (dist * 1000 )
            tokens += ' '.join([''.join(word) for _ in range(word_occurance)])
            tokens += ' '

        wordcloud = WordCloud(width=800, height=560,
                              background_color='white', collocations=False,
                              min_font_size=10).generate(tokens)

    else:
        return

    # plot the WordCloud image
    plt.figure(figsize=(8, 5.6), facecolor=None)
    plt.imshow(wordcloud)
    plt.axis("off")
    plt.tight_layout(pad=0)

    plt.savefig(dir + 'Topic' + str(topic) + '_wordcloud')
    print('Getting wordcloud for topic {}. Done!'.format(topic))


def get_employee_clusters(averaged_employee_features, cluster_titles ,employee_titles, vis = 'Employee Cluster 2D', dir = os.getcwd()):


    fig = go.Figure()
    k = len(cluster_titles)
    if vis == 'Employee Cluster 2D':
        for i, cl in enumerate(range(k)):
            cluster_df = averaged_employee_features[averaged_employee_features.cluster == cl]

            # plot all the employees belonging to a cluster using a new trace.
            features = np.stack(cluster_df['feature_vector_2d'].to_numpy())

            # Normalize the features between 0 and 1 before visualization
            min_ = features.min(axis=0)
            max_ = features.max(axis=0)
            features = (features - min_) / (max_ - min_)

            # constructing hover text for all employees belonging to that cluster
            employee_titles_this_cluster = [employee_titles[emp] for emp in cluster_df.index.values]

            fig.add_trace(go.Scattergl(
                x=features[:, 0],
                y=features[:, 1],
                mode='markers',
                showlegend=True,
                text=employee_titles_this_cluster,
                marker=dict(size=8, color=i),
                name="cluster_" + str(cl + 1) + ": " + cluster_titles[cl]
            ))

        fig.update_layout(title='Distribution of employees over clusters in 2d space', legend=dict(
            x=0.6,
            y=1.4,
            traceorder='normal',
            font=dict(
                size=10)))
        path = dir + '2D_clusters.jpeg'
    else:
        #vis is 3d (Employee Cluster 3D)
        fig = go.Figure()

        for i, cl in enumerate(range(k)):
            cluster_df = averaged_employee_features[averaged_employee_features.cluster == cl]

            # plot all the employees belonging to a cluster using a new trace.
            features = np.stack(cluster_df['feature_vector_3d'].to_numpy())

            # Normalize the features between 0 and 1 before visualization
            min_ = features.min(axis=0)
            max_ = features.max(axis=0)

            features = (features - min_) / (max_ - min_)

            # constructing hover text for all employees belonging to that cluster
            employee_titles_this_cluster = [employee_titles[emp] for emp in cluster_df.index.values]
            print (employee_titles_this_cluster)

            fig.add_trace(go.Scatter3d(
                x=features[:, 0],
                y=features[:, 1],
                z=features[:, 2],
                mode='markers',
                showlegend=True,
                text=employee_titles_this_cluster,
                marker=dict(size=3, color=i),
                name="cluster_" + str(cl + 1) + ": " + cluster_titles[cl]
            ))

        fig.update_layout(title='Distribution of employees over clusters in 3d Space', legend=dict(
            x=0.6,
            y=1.4,
            traceorder='normal',
            font=dict(
                size=10)))
    path = dir + '3D_clusters.jpeg'

    fig.show()
    fig.write_image(path)
    print ("Saved Clusters to " , path)

def create_hist(employee_cluster_frequency):

    paths = []

    '''
    1. Bar chart for Employee distribution per cluster
    '''
    y = list(employee_cluster_frequency)
    x = ['Cluster {}'.format(str(i+1)) for i in range(len(y))]

    fig = go.Figure(data=[go.Bar(x=x , y=y)])
    fig.update_layout(
        title_text='Employee Distribution per cluster',  # title of plot
        xaxis_title_text='Clusters',  # xaxis label
        yaxis_title_text='Employees percentage',  # yaxis label

    )
    fig.show()

    #hist_path = 'flaskApp/static/quick_stat.jpeg'
    #paths.append("quick_stat.jpeg")
    #fig.write_image(hist_path)

    '''
    2. Further quick stats charts
    '''
    #return paths





