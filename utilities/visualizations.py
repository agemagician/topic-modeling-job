import umap
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import numpy as np
import os

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

