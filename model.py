from sklearn.decomposition import LatentDirichletAllocation as LDA

from visualization import plot_10_most_common_words


def _print_topics(model, words, number_words):
    print("Topics found via LDA:")
    for topic_idx, topic in enumerate(model.components_):
        print("\nTopic #%d:" % topic_idx)
        print(" ".join([words[i]
                        for i in topic.argsort()[:-number_words - 1:-1]]))


def apply_LDA(words, paper_words_count_matrix, number_topics, number_words):

    # Create and fit the LDA model
    lda = LDA(n_components=number_topics)
    lda.fit(paper_words_count_matrix)

    # Print the topics found by the LDA model
    _print_topics(lda, words, number_words)

    return lda


def get_words_count_matrix(papers, count_vectorizer):
    # Fit and transform the processed titles
    paper_words_count_matrix = count_vectorizer.fit_transform(papers['paper_text_processed'])

    # Visualise the 10 most common words
    plot_10_most_common_words(paper_words_count_matrix, words=count_vectorizer.get_feature_names())

    return paper_words_count_matrix
