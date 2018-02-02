from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import NMF
from sklearn.base import TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.stop_words import ENGLISH_STOP_WORDS

from nltk import sent_tokenize
from nltk.corpus import stopwords

from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

from sqlalchemy import create_engine
from sqlalchemy_utils import database_exists, create_database
import psycopg2

import pandas as pd
import numpy as np

import string
import spacy

#------------------------------------------------------------------
def unique(sequence):
    '''get unique elements of list and keep the same order'''

    seen = set()
    return [x for x in sequence if not (x in seen or seen.add(x))]
#------------------------------------------------------------------


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
def return_topics(vectorizer, clf, W, df, n_top_words, n_top_documents):
    ''' Return topics discovered by a model '''

    # get list of feature names
    feature_names = vectorizer.get_feature_names()

    # get VADER sentiment analyzer
    analyser      = SentimentIntensityAnalyzer()

    # list of topics, polarities  and reviews to return
    topics, reviews = [], []

    # loop over all the topics
    for topic_id, topic in enumerate(clf.components_):

        # grab the list of words describing the topic
        word_list = []
        for i in topic.argsort()[:-n_top_words - 1:-1]:
            word_list.append(feature_names[i])

        # split words in case there are some bigrams and get unique set
        split_list = []
        for word in word_list:
            for split in word.split():
                split_list.append(split)
        topic_words = list(set(split_list))

        # append topic words as a single string
        topics.append(' '.join([word for word in topic_words]))

        # loop over reviews for each topic
        top_doc_indices = np.argsort( W[:,topic_id] )[::-1][0:n_top_documents]
        for doc_index in top_doc_indices:

            # check that the review contains one of the topic words
            review = df['reviewText'].iloc[doc_index]
            if any(word in review.lower() for word in topic_words):

                # seniment analysis
                vader = analyser.polarity_scores(review)

                # append current review with seniment and topic id to the list
                reviews.append(df.iloc[doc_index].to_dict())
                reviews[-1]['topic']     = topic_id
                reviews[-1]['sentiment'] = vader['compound']

    return topics, reviews
#------------------------------------------------------------------------------

#------------------------------------------------------------------------------
def summarize_reviews(topics, reviews):
    ''' extract relevant sentences from a review for this topic
         do sentiment analysis just for those sentences '''

    # define sentiment analyzer
    analyser = SentimentIntensityAnalyzer()

    # loop over reviews and summarize content
    for i, review in enumerate(reviews):
        summary     = []
        sentences   = sent_tokenize(review['reviewText'])
        topic_words = topics[review['topic']].split()
        for sentence in sentences:
            if any(word in sentence.lower() for word in topic_words):
                highlighted_sentence = '<span style="background-color: #FFFF00">'+sentence+'</span>'
                summary.append(highlighted_sentence)
            else:
                summary.append(sentence)
        print(summary)

        # save info for summarized reviews
        reviews[i]['summarized_reviewText'] = ' '.join([sent for sent in summary])
        vader = analyser.polarity_scores(reviews[i]['summarized_reviewText'])
        reviews[i]['summary_sentiment']     = vader['compound']

        # add html for user rating
        rating_html = ''
        rating = int(reviews[i]['overall'])
        for _ in range(rating):
            rating_html += '<span class="fa fa-star checked"></span>'
            for _ in range(rating,5):
                rating_html += '<span class="fa fa-star"></span>'
        reviews[i]['overall_html'] = rating_html

    return reviews
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


    # define the name of the database
    dbname = 'amazon_reviews'
    username = 'plestran'
    engine = create_engine('postgres://%s@localhost/%s'%(username,dbname))

    ## create a database (if it doesn't exist)
    if not database_exists(engine.url):
        create_database(engine.url)

    # Connect to make queries using psycopg2
    con = psycopg2.connect(database = dbname, user = username)

    # grab reviews for current product
    sql_query = """
        SELECT * FROM reviews
        WHERE asin = '%s';
        """ % test_asin
    test_df = pd.read_sql_query(sql_query,con)

    #reviews_df = getDF('data/reviews_Electronics_5_first1000.json')
#    test_asin  = reviews_df['asin'].value_counts().idxmax()
    #test_df    = reviews_df[reviews_df['asin'] == test_asin].dropna()

    # define the number features, topics, and how many
    # words/documents to display later on
    n_features      = 1000
    n_topics        = min(int(test_df['reviewText'].size/2),6)
    n_top_words     = 3
    n_top_documents = min(int(test_df['reviewText'].size/2),3)

    # Use tf-idf vectorizer
    vectorizer = TfidfVectorizer(max_features=n_features,
                                 tokenizer=tokenizeText,
                                 stop_words='english',
                                 ngram_range=(1,2),
                                 max_df=0.9, min_df=3)

    # use NMF model with the Frobenius norm
    clf = NMF(n_components=n_topics, random_state=1,
              solver='mu', beta_loss='frobenius')

    # put it all in a pipeline
    pipe = Pipeline([('cleanText', CleanTextTransformer()),
                     ('vectorizer', vectorizer), ('nmf', clf)])

    # Fit the model
    pipe.fit(test_df['reviewText']);

    # grab term-document matrix
    transform = pipe.fit_transform(test_df['reviewText'])

    # grab the topic words and avg polarities from the model
    topics, reviews = return_topics(vectorizer, clf, transform, test_df,
                                    n_top_words, n_top_documents)

    # summarize reviews
    reviews = summarize_reviews(topics, reviews)
    print(reviews)

    return topics, reviews
#------------------------------------------------------------------------------

#------------------------------------------------------------------------------
def in_db(asin):

    # define the name of the database
    dbname = 'amazon_reviews'
    username = 'plestran'
    engine = create_engine('postgres://%s@localhost/%s'%(username,dbname))

    # Connect to make queries using psycopg2
    con = psycopg2.connect(database = dbname, user = username)

    # grab reviews for current product
    sql_query = """
        SELECT * FROM reviews
        WHERE asin = '%s';
        """ % asin
    reviews_df = pd.read_sql_query(sql_query,con)

    return asin in reviews_df['asin'].tolist()
#------------------------------------------------------------------------------
