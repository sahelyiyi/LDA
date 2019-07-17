import sys
from collections import defaultdict
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation as LDA
import numpy as np

from data.papers_data import load_data
from models.lda import print_lda_topics, lda_topics_visualization
from models.rtm import RTM, print_rtm_topics
from settings import NUMBER_TOPICS, NUMBER_WORDS, RTM_MAX_ITER


def run_LDA(count_vectorizer, words, paper_words_count_matrix, number_topics=NUMBER_TOPICS, number_words=NUMBER_WORDS):
    print ('Running LDA')

    # Create LDA models
    model = LDA(n_components=number_topics)

    # Fit LDA models
    model.fit(paper_words_count_matrix)
    print ('LDA models has fitted.')

    print_lda_topics(model, words, number_words)

    # Visualising topics
    lda_topics_visualization(model, count_vectorizer, paper_words_count_matrix, number_topics)
    print ('LDA topics visualizations has stored.')


def run_RTM(vocab, paper_words_count_matrix, max_iter=RTM_MAX_ITER, number_topics=NUMBER_TOPICS, number_words=NUMBER_WORDS):
    print ('Running RTM')

    # Create RTM models
    number_docs = paper_words_count_matrix.shape[0]
    number_vocab = len(vocab)
    model = RTM(number_topics, number_docs, number_vocab, verbose=True)

    # Fit RTM models
    doc_links = defaultdict(list)
    model.fit(paper_words_count_matrix, doc_links, max_iter=max_iter)
    print ('RTM models has fitted.')

    print_rtm_topics(model, vocab, number_topics, number_words)

    # Save models
    model.save_model()
    print ('Saved models parameters.')


def run(method_name='LDA'):
    # Load papers data
    papers = load_data()
    print ('Data has loaded.')

    # Initialise the count vectorizer with the English stop words
    count_vectorizer = CountVectorizer(stop_words='english')

    # Prepare matrix data
    paper_words_count_matrix = count_vectorizer.fit_transform(papers['paper_text_processed'])
    words = np.array(count_vectorizer.get_feature_names())
    print ('Documents matrix has prepared.')

    if method_name == 'LDA':
        run_LDA(count_vectorizer, words, paper_words_count_matrix)
    elif method_name == 'RTM':
        run_RTM(words, paper_words_count_matrix)


run(sys.argv[1])
