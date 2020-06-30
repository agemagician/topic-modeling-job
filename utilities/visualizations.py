from collections import Counter
import umap
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import numpy as np
import os


def plot_proj(embedding, lbs):
    """
    Plot UMAP embeddings
    :param embedding: UMAP (or other) embeddings
    :param lbs: labels
    """
    n = len(embedding)
    counter = Counter(lbs)
    for i in range(len(np.unique(lbs))):
        plt.plot(embedding[:, 0][lbs == i], embedding[:, 1][lbs == i], '.', alpha=0.5,
                 label='cluster {}: {:.2f}%'.format(i, counter[i] / n * 100))
    plt.legend(loc = 'best')
    plt.grid(color ='grey', linestyle='-',linewidth = 0.25)



def get_wordcloud(model, token_lists=None, topic=None, word_distribution=None):
    """
    Get word cloud of each topic from fitted model
    :param model: Topic_Model object
    :param sentences: preprocessed sentences from docs
    """

    print('Getting wordcloud for topic {} ...'.format(topic))
    if word_distribution is not None:
        tokens = str()
        for item in word_distribution:
            word = item[0]
            dist = item[1]
            # from porobability to actually numbers the word appears in the document
            word_occurance = int (dist * 1000 )
            tokens += ' '.join([''.join(word) for _ in range(word_occurance)])
            tokens += ' '
        dr = 'output/{}/{}/'.format("LDA", model.id)
            
    else:
        lbs = model.cluster_model.labels_
        tokens = ' '.join([' '.join(_) for _ in np.array(token_lists)[lbs == topic]])
        dr = 'output/{}/{}/'.format(model.method, model.id)
    
    print (tokens)
    wordcloud = WordCloud(width=800, height=560,
                          background_color='white', collocations=False,
                          min_font_size=10).generate(tokens)

    # plot the WordCloud image
    plt.figure(figsize=(8, 5.6), facecolor=None)
    plt.imshow(wordcloud)
    plt.axis("off")
    plt.tight_layout(pad=0)
    if not os.path.exists(dr):
        os.makedirs(dr)
    plt.savefig(dr + 'Topic' + str(topic) + '_wordcloud')
    print('Getting wordcloud for topic {}. Done!'.format(topic))

def visualize(model):
    """
    Visualize the result for the topic model by 2D embedding (UMAP)
    :param model: Topic_Model object
    """
    reducer = umap.UMAP()
    print('Calculating UMAP projection ...')
    vec_umap = reducer.fit_transform(model.vec[model.method])
    print('Calculating UMAP projection. Done!')
    plot_proj(vec_umap, model.cluster_model.labels_)
    dr = 'output/{}/{}'.format(model.method, model.id)
    print (dr)
    if not os.path.exists(dr):
        os.makedirs(dr)
    plt.savefig('output/{}/{}/2dvis.png'.format(model.method, model.id))