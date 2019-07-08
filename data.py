import re

import pandas as pd

from visualization import visualizing_papers_word_cloud


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
    visualizing_papers_word_cloud(papers)

    return papers
