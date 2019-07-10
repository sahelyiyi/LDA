import re

import pandas as pd
from wordcloud import WordCloud


def _clean_data(papers):
    # Remove punctuation
    papers['paper_text_processed'] = papers['paper_text'].map(lambda x: re.sub('[,\.!?]', '', x))

    # Convert the titles to lowercase
    papers['paper_text_processed'] = papers['paper_text_processed'].map(lambda x: x.lower())


def loading_data():
    # Read data into papers
    papers = pd.read_csv('papers.csv')

    # Remove the columns
    papers = papers.drop(columns=['id', 'event_type', 'pdf_name'], axis=1)

    # Cleaning text
    _clean_data(papers)

    # visualizing data
    # visualizing_papers_word_cloud(papers)

    return papers


def visualizing_papers_word_cloud(papers):
    # Join the different processed titles together.
    long_string = ','.join(list(papers['paper_text_processed'].values))

    # Create a WordCloud object
    wordcloud = WordCloud(background_color="white", max_words=5000, contour_width=3, contour_color='steelblue')

    # Generate a word cloud
    wordcloud.generate(long_string)

    # Visualize the word cloud
    wordcloud.to_image().show()
