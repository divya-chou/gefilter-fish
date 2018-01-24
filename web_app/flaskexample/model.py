from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import NMF
from sklearn.base import TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.stop_words import ENGLISH_STOP_WORDS
from sklearn.metrics import accuracy_score
from nltk.corpus import stopwords
from textblob import TextBlob

import pandas as pd
import numpy as np

import string
import spacy
import re

#------------------------------------------------------------------------------
def parse(path):
    g = open(path, 'rb')
    for l in g:
        yield eval(l)

def getDF(path):

    i = 0
    df = {}
    for d in parse(path):
        df[i] = d
        i += 1
    return pd.DataFrame.from_dict(df, orient='index')
#------------------------------------------------------------------------------

#------------------------------------------------------------------------------
class CleanTextTransformer(TransformerMixin):
    """
    Convert text to cleaned text
    """

    def transform(self, X, **transform_params):
        return [cleanText(text) for text in X]

    def fit(self, X, y=None, **fit_params):
        return self

    def get_params(self, deep=True):
        return {}
#------------------------------------------------------------------------------

#------------------------------------------------------------------------------
def cleanText(text):
    ''' A custom function to clean the text before sending it into the
        vectorizer
    '''

    # import a dictionary of English contractions from another file
    from contractions import english_contractions
    contraction_dict = english_contractions()

    # replace the contractions with their expanded form
    for contraction, expansion in contraction_dict.items():
        text = text.replace(contraction.lower(),expansion.lower())

    # get rid of newlines
    text = text.strip().replace("\n", " ").replace("\r", " ")

    # lowercase
    text = text.lower()

    return text
#------------------------------------------------------------------------------

#------------------------------------------------------------------------------
def tokenizeText(sample):
    ''' A custom function to tokenize the text using spaCy
        and convert to lemmas '''

    # get the tokens using spaCy
    tokens = PARSER(sample)

    # lemmatize
    lemmas = []
    for tok in tokens:
        lemmas.append(tok.lemma_.lower().strip()
                      if tok.lemma_ != "-PRON-" else tok.lower_)
    tokens = lemmas

    # stoplist the tokens
    tokens = [tok for tok in tokens if tok not in STOPLIST]

    # stoplist symbols
    tokens = [tok for tok in tokens if tok not in SYMBOLS]

    # remove large strings of whitespace
    while "" in tokens:
        tokens.remove("")
    while " " in tokens:
        tokens.remove(" ")
    while "\n" in tokens:
        tokens.remove("\n")
    while "\n\n" in tokens:
        tokens.remove("\n\n")

    return tokens
#------------------------------------------------------------------------------

#------------------------------------------------------------------------------
def return_topics(vectorizer, nmf, W, df, n_top_words, n_top_documents):
    ''' Return topics discovered by a model '''

    feature_names = vectorizer.get_feature_names()

    topics, polarities, reviews = [], [], []
    for topic_id, topic in enumerate(nmf.components_):
        # grab topic words
        topics.append(' '.join([str(feature_names[i])
                      for i in topic.argsort()[:-n_top_words - 1:-1]]))

        # grab average polarities
        top_doc_indices = np.argsort(W[:,topic_id])[::-1][0:n_top_documents]
        avg_polarity = 0
        for doc_index in top_doc_indices:
            doc = TextBlob(df['reviewText'].iloc[doc_index])
            avg_polarity += doc.sentiment[0]
        avg_polarity /= n_top_documents
        if avg_polarity <= -0.3:
            polarities.append("Bad")
        elif avg_polarity > -0.3 and avg_polarity <= 0.3:
            polarities.append("Neutral")
        else:
            polarities.append("Good")

        # return indices of documents in each topic
        for doc_index in top_doc_indices:
            reviews.append(df.iloc[doc_index].to_dict())
            reviews[-1]['topic'] = topic_id

    return topics, polarities, reviews
#------------------------------------------------------------------------------

#------------------------------------------------------------------------------
def print_topics(test_asin):

    global PARSER, STOPLIST, SYMBOLS

    PARSER = spacy.load('en')

    # A custom stoplist
    STOPLIST = set(stopwords.words('english') + list(ENGLISH_STOP_WORDS))
    # List of symbols we don't care about
    SYMBOLS = " ".join(string.punctuation).split(" ") + \
              ["-----", "---", "...", "“", "”", "'s"]

    reviews_df = getDF('/Users/plestran/Dropbox/insight/gefilter-fish/data/reviews_Electronics_5_first1000.json')
#    test_asin  = reviews_df['asin'].value_counts().idxmax()
    test_df    = reviews_df[reviews_df['asin'] == test_asin].dropna()

    # define the number features, topics, and how many
    # words/documents to display later on
    n_features      = 1000
    n_topics        = min(int(test_df['reviewText'].size/2),10)
    n_top_words     = 3
    n_top_documents = min(int(test_df['reviewText'].size/2),5)

    # Use tf-idf vectorizer
    vectorizer = TfidfVectorizer(max_features=n_features,
                                 tokenizer=tokenizeText,
                                 stop_words='english')

    # use NMF model with the Frobenius norm
    nmf = NMF(n_components=n_topics, random_state=1,
              solver='mu', beta_loss='frobenius')

    # put it all in a pipeline
    pipe = Pipeline([('cleanText', CleanTextTransformer()),
                     ('vectorizer', vectorizer), ('nmf', nmf)])

    # Fit the model
    pipe.fit(test_df['reviewText']);

    # grab term-document matrix
    transform = pipe.fit_transform(test_df['reviewText'])

    # grab the topic words and avg polarities from the model
    topics, polarities, reviews = return_topics(vectorizer, nmf, transform,
                                                test_df, n_top_words, n_top_documents)

    return topics, polarities, reviews
#------------------------------------------------------------------------------

#------------------------------------------------------------------------------
def in_db(asin):

    reviews_df = getDF('/Users/plestran/Dropbox/insight/gefilter-fish/data/reviews_Electronics_5_first1000.json')

    return asin in reviews_df['asin'].tolist()
#------------------------------------------------------------------------------
