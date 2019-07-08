
from sklearn.feature_extraction.text import CountVectorizer

from data import loading_data
from model import get_words_count_matrix, apply_LDA
from settings import NUMBER_TOPICS, NUMBER_WORDS
from visualization import topics_visualization


def run(number_topics=NUMBER_TOPICS, number_words=NUMBER_WORDS):
    # Load papers data
    papers = loading_data()
    print ('Pass loading papers data.')

    # Initialise the count vectorizer with the English stop words
    count_vectorizer = CountVectorizer(stop_words='english')

    # Prepare LDA matrix data
    paper_words_count_matrix = get_words_count_matrix(papers, count_vectorizer)
    print ('Pass prepare LDA matrix data.')

    # Apply LDA
    words = count_vectorizer.get_feature_names()
    lda = apply_LDA(words, paper_words_count_matrix, number_topics, number_words)
    print ('Pass apply LDA.')

    # Visualising topics
    topics_visualization(lda, count_vectorizer, paper_words_count_matrix, number_topics)
    print ('Pass store topics visualization.')


run()
