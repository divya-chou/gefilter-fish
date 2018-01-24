from flask import render_template
from flask import request
from flask import jsonify

from flaskexample import app
from flaskexample.model import print_topics, in_db

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

    topics, polarities, reviews = print_topics("0972683275")

    outputs = []
    for topic, polarity in zip(topics, polarities):
        outputs.append(dict(topic=topic, polarity=polarity))

    return render_template('topics.html',outputs=outputs)
#------------------------------------------------------------------------------

#------------------------------------------------------------------------------
@app.route('/reviews', methods=['POST'])
def reviews_page():
    import pandas as pd

    # grab product information
    topic_id = int(request.args.to_dict()['topic'])
    title    = request.args.to_dict()['title']
    product  = dict(productName=title,
                    topicWords=topics[topic_id])

    # grab the reviews for this topic
    reviews_df = pd.DataFrame(reviews)
    reviews_df = reviews_df[reviews_df['topic'] == topic_id]
    relevant_reviews = reviews_df.T.to_dict().values()

    return render_template('reviews.html', product=product,
                           reviews=relevant_reviews)
#------------------------------------------------------------------------------

#------------------------------------------------------------------------------
@app.route('/model', methods=['GET'])
def return_topics():
    '''Return topics extracted from reviews in our database'''

    global topics, polarities, reviews

    asin = request.args.get("asin")
    topics, polarities, reviews = print_topics(asin)

    # append emojis showing whether the sentiment
    emoji_topics = []
    for i, polarity in enumerate(polarities):
        if polarity == "Bad":
            emoji_topics.append("\uD83D\uDC4E - "+topics[i])
        elif polarity == "Neutral":
            emoji_topics.append("\uD83D\uDE10 - "+topics[i])
        else:
            emoji_topics.append("\uD83D\uDC4D - "+topics[i])

    return jsonify(topic=emoji_topics)
#------------------------------------------------------------------------------

#------------------------------------------------------------------------------
@app.route('/in_db', methods=['GET'])
def return_in_db():
    '''Check if an ASIN is in our database'''

    asin = request.args.get("asin")
    return jsonify( in_db( asin ) )
#------------------------------------------------------------------------------
