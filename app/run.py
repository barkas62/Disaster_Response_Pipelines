import json
import joblib
import numpy as np
import plotly
import pandas as pd

import re
import nltk
#from nltk.stem import WordNetLemmatizer
#from nltk.tokenize import word_tokenize

from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.stem.porter import PorterStemmer

nltk.download('stopwords')
nltk.download('wordnet')

from flask import Flask
from flask import render_template, request, jsonify
from plotly.graph_objs import Bar, Pie
from sqlalchemy import create_engine


app = Flask(__name__)

def tokenize(text):
    # Normalize text: leave decapitalized letters and digits only
    text = re.sub(r"[^a-zA-Z0-9]", " ", text.lower())

    # Split into words
    words = text.split()

    # Remove stop words
    words = [w for w in words if w not in stopwords.words("english")]

    # Reduce words to their root form
    lemmed = [WordNetLemmatizer().lemmatize(w) for w in words]

    # Reduce words to their stems
    stemmed = [PorterStemmer().stem(w) for w in lemmed]

    return stemmed


# load data
engine = create_engine('sqlite:///../data/DisasterResponse.db')
df = pd.read_sql_table('labeled_messages', engine)
engine.dispose()

# load model
model = joblib.load("../models/disaster_model.pkl")


# index webpage displays cool visuals and receives user input text for model
@app.route('/')
@app.route('/index')
def index():

    # extract data needed for visuals

    lengths = df.message.str.split().str.len()
    length_counts, length_division = np.histogram(lengths,
                                              range=(0, lengths.quantile(0.99)))

    translated_names = ['Translated Messages', 'Messages in English']
    translated_counts = [df['original'].notnull().sum(), df['original'].isnull().sum()]

    category_names = df.columns[4:].tolist()
    category_counts = df[category_names].sum().sort_values(ascending=False)

    # create visuals
    graphs = [
        {
            'data': [
                Bar(
                    x=length_division,
                    y=length_counts
                    )
            ],

            'layout': {
                'title': 'Message Length Distribution',
                },
            'yaxis': {
                'title': "Count"
            },
            'xaxis': {
                'title': "Message Length"
            }
        },
        {
            'data': [
                Pie(
                    labels=translated_names,
                    values=translated_counts
                )
            ],

            'layout': {
                'title': 'Percentage of Messages in English and Translated',
            }
        },
        {
            'data': [
                Bar(
                    x=category_names,
                    y=category_counts
                )
            ],

            'layout': {
                'title': 'Distribution of Message Categories',
                'yaxis': {
                    'title': "Message Count"
                },
                'xaxis': {
                    'title': "Category Name"
                }
            }
        }
    ]

    # encode plotly graphs in JSON
    ids = ["graph-{}".format(i) for i, _ in enumerate(graphs)]
    graphJSON = json.dumps(graphs, cls=plotly.utils.PlotlyJSONEncoder)

    # render web page with plotly graphs
    return render_template('master.html', ids=ids, graphJSON=graphJSON)


# web page that handles user query and displays model results
@app.route('/go')
def go():
    # save user input in query
    query = request.args.get('query', '')

    # use model to predict classification for query
    classification_labels = model.predict([query])[0]

    print( classification_labels )

    classification_results = dict(zip(df.columns[4:], classification_labels))

    # This will render the go.html Please see that file.
    return render_template(
        'go.html',
        query=query,
        classification_result=classification_results
    )


def main():
    app.run(host='0.0.0.0', port=3001, debug=True)


if __name__ == '__main__':
    main()
