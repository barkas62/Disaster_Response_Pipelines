import re
import sys

import pandas as pd
from sqlalchemy import create_engine

import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from nltk.stem.wordnet import WordNetLemmatizer

nltk.download('stopwords')
nltk.download('wordnet')

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.multioutput import MultiOutputClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import GridSearchCV

from sklearn.decomposition import TruncatedSVD
from sklearn.neighbors import KNeighborsClassifier

import joblib

def load_data(database_filepath):
    '''
    Load data from sqlight database; gets category names
    :param database_filepath: string : filepath of sqlight database
    :return: explanatory variables X (DataSeries) and targets Y (DataFrame), plus names of categories
    '''
    engine = create_engine('sqlite:///' + database_filepath)
    df = pd.read_sql_table('messages_categories', engine)
    engine.dispose()

    # Names of ctategories
    categories_cols = df.columns[4:].tolist()

    X = df['message']
    Y = df[categories_cols]

    return X, Y, categories_cols


def tokenize(text):
    '''
    Convert text into list of tokens
    :param text: string : a text
    :return: list of tokens
    '''
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


def build_model():
    '''
    Build Pipeline and GridSearchCV objects
    :param : None
    :return: Grid search scikit-learn GridSearchCV model object
    '''

    knn_classifier = KNeighborsClassifier()

    pipeline_knn = Pipeline([
        ('tfidf', TfidfVectorizer(tokenizer=tokenize)),
        ('truncate', TruncatedSVD(n_components=32)),
        ('knn', MultiOutputClassifier(knn_classifier))
    ])

    parameters_knn = {
        'tfidf__ngram_range': ((1, 1), (1, 2)),
        'tfidf__max_df': (0.75, 1.0),
        'truncate__n_components': [16, 32, 64],
        'knn__estimator__n_neighbors': [1, 3]
    }

    knn_cv = GridSearchCV(pipeline_knn, parameters_knn, cv=3, verbose=3)

    return knn_cv


def evaluate_model(model, X_test, Y_test, category_names):
    '''
    Generate predictions for a test set of explanatory variables;
    Compute quality metrics by comparing predicted and predefined target data
    :param model: sklearn estimator : fitted estimator object
    :param X_test: DataSeries : Test set of text messages
    :param Y_test: DataFrame  : predefined target categories data
    :param category_names: names of columns in Y_test, containing target categories
    :return: None
    '''
    # Generate predictions
    Y_pred = model.predict(X_test)

    # Print average scores
    print('Accuracy: ', accuracy_score(Y_test, Y_pred))
    print('Precision: ', precision_score(Y_test, Y_pred, average='weighted'))
    print('Recall: ', recall_score(Y_test, Y_pred, average='weighted'))
    print('F1 score: ', f1_score(Y_test, Y_pred, average='weighted'))

    # Print detailed  report
    print(classification_report(Y_test, Y_pred, target_names=category_names))


def save_model(model, model_filepath):
    '''
    Dumps the model to pickle file with a given filepath
    :param model: sklearn model : model to dump
    :param model_filepath: string : filepath of pickle file
    :return: None
    '''

    joblib.dump(model, model_filepath)


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        
        print('Building model...')
        model = build_model()
        
        print('Training model...')
        model.fit(X_train, Y_train)
        
        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()