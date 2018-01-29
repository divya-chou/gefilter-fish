from flask import render_template
from flask import request
from flask import jsonify

from flaskexample import app
from flaskexample.model import print_topics, in_db, unique

import pandas as pd

#------------------------------------------------------------------------------
@app.route('/')
@app.route('/index')
def index():
    return render_template("index.html",
       title = 'Gefilter Fish', user = { 'nickname': 'Insight' },
       )
#------------------------------------------------------------------------------

#------------------------------------------------------------------------------
@app.route('/topics', methods=['POST'])
def topic_page():
    ''' Display an example of topics extracted from some reviews '''

    topics, reviews = print_topics("0972683275")

    outputs = []
    for topic, review in zip(topics, reviews):
        outputs.append(dict(topic=topic, polarity=review['sentiment']))

    return render_template('topics.html',outputs=outputs)
#------------------------------------------------------------------------------

#------------------------------------------------------------------------------
@app.route('/reviews', methods=['POST'])
def reviews_page():
    ''' Render page showing summarized reviews '''

    # grab product information
    topic_id = int(request.args.to_dict()['topic'])
    title    = request.args.to_dict()['title']
    product  = dict(productName=title, topicWords=topics[topic_id])

    # grab the reviews for this topic
    reviews_df = pd.DataFrame(reviews)
    reviews_df = reviews_df[reviews_df['topic'] == topic_id]
    relevant_reviews = reviews_df.T.to_dict().values()

    return render_template('reviews.html', product=product,
                           reviews=relevant_reviews)
#------------------------------------------------------------------------------

#------------------------------------------------------------------------------
@app.route('/model', methods=['GET'])
def display_topics():
    ''' Return topics extracted from reviews in our database '''

    global topics, reviews

    # get asin, topics, and reviews
    asin = request.args.get("asin")
    if asin == '0972683275':
        reviews = pd.read_csv('/home/ubuntu/application/flaskexample/0972683275_reviews.csv')
        topics  = unique(reviews['topic_words'])
        reviews = list(reviews.T.to_dict().values())
    else:
        topics, reviews = print_topics(asin)

    # append emojis showing the sentiment
    emoji_topics = []
    for topic_id, topic in enumerate(topics):

        # grab the average sentiment
        reviews_df = pd.DataFrame(reviews)
        reviews_df = reviews_df[reviews_df['topic'] == topic_id]
        polarity   = reviews_df['summary_sentiment'].mean()

        # append emojis to topic name based on range of sentiment
        if polarity <= -0.5:
            emoji_topics.append("\uD83D\uDC4E - "+topic)
        elif polarity > -0.5 and polarity < 0.5:
            emoji_topics.append("\uD83D\uDE10 - "+topic)
        else:
            emoji_topics.append("\uD83D\uDC4D - "+topic)

    return jsonify(topic=emoji_topics)
#------------------------------------------------------------------------------

#------------------------------------------------------------------------------
@app.route('/in_db', methods=['GET'])
def return_in_db():
    ''' Check if an ASIN is in our database '''

    asin = request.args.get("asin")
    return jsonify( in_db( asin ) )
#------------------------------------------------------------------------------
