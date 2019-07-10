import os
import pickle
import pyLDAvis

from pyLDAvis import sklearn as sklearn_lda

from settings import LDA_RESULTS_FILE_PATH


def print_lda_topics(model, words, number_words):
    print("Topics found via LDA:")
    for topic_idx, topic in enumerate(model.components_):
        print("\nTopic #%d:" % topic_idx)
        print(" ".join([words[i]
                        for i in topic.argsort()[:-number_words - 1:-1]]))


def lda_topics_visualization(lda, count_vectorizer, paper_words_count_matrix, number_topics):
    results_file_path = os.path.join(LDA_RESULTS_FILE_PATH, 'ldavis_prepared_' + str(number_topics))
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
