import os
import pickle
import pyLDAvis

import numpy as np
import seaborn as sns

from matplotlib import pyplot as plt
from pyLDAvis import sklearn as sklearn_lda
from wordcloud import WordCloud

from settings import RESULTS_FILE_PATH


def plot_10_most_common_words(paper_words_count_matrix, words):
    total_words_counts = np.zeros(len(words))
    for t in paper_words_count_matrix:
        total_words_counts += t.toarray()[0]

    sorted_top_words_counts_tuple = sorted(zip(words, total_words_counts), key=lambda x: x[1], reverse=True)[0:10]
    words = [w[0] for w in sorted_top_words_counts_tuple]
    words_counts = [w[1] for w in sorted_top_words_counts_tuple]

    x_pos = np.arange(len(words))

    plt.figure(2, figsize=(15, 15 / 1.6180))
    plt.subplot(title='10 most common words')
    sns.set_context("notebook", font_scale=1, rc={"lines.linewidth": 2})
    sns.barplot(x_pos, words_counts, palette='husl')
    plt.xticks(x_pos, words, rotation=90)
    plt.xlabel('words')
    plt.ylabel('counts')
    plt.show()


def topics_visualization(lda, count_vectorizer, paper_words_count_matrix, number_topics):
    results_file_path = os.path.join(RESULTS_FILE_PATH, 'ldavis_prepared_' + str(number_topics))
    # # this is a bit time consuming - make the if statement True
    # # if you want to execute visualization prep yourself
    if 1 == 1:
        LDAvis_prepared = sklearn_lda.prepare(lda, paper_words_count_matrix, count_vectorizer)
    with open(results_file_path, 'wb') as f:
        pickle.dump(LDAvis_prepared, f)

    # load the pre-prepared pyLDAvis data from disk
    with open(results_file_path, 'rb') as f:
        LDAvis_prepared = pickle.load(f)
    pyLDAvis.save_html(LDAvis_prepared, results_file_path + '.html')


def visualizing_papers_word_cloud(papers):
    # Join the different processed titles together.
    long_string = ','.join(list(papers['paper_text_processed'].values))

    # Create a WordCloud object
    wordcloud = WordCloud(background_color="white", max_words=5000, contour_width=3, contour_color='steelblue')

    # Generate a word cloud
    wordcloud.generate(long_string)

    # Visualize the word cloud
    wordcloud.to_image().show()
