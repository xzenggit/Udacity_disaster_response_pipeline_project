import sys
from sqlalchemy import create_engine
import pandas as pd
import pickle

from sklearn.pipeline import Pipeline
from sklearn.multioutput import MultiOutputClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report

import nltk
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

nltk.download(['punkt', 'wordnet'])

def load_data(database_filepath):
    """
    Load data from database file.
    Args:
        database_filepath: file directory and name of database
    Return:
        X: model input
        Y: model output
        
    """
    # load data from database
    engine = create_engine('sqlite:///'+database_filepath)
    df = pd.read_sql_table('DisasterResponse', engine)
    X = df['message']
    Y = df.iloc[:, 3:]
    category_names = Y.columns
    
    return X, Y, category_names


def tokenize(text):
    """
    Text tokenizer
    
    Args: 
        text: text to be tokenized.
    Return:
        clean_tokens: tokenized text.
    """
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()
    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)

    return clean_tokens


def build_model():
    """
    Build the machine learning model.

    """
    
    # Create pipeline
    pipeline = Pipeline([
        ('vectorize', CountVectorizer(tokenizer=tokenize)),
        ('tf-idf', TfidfTransformer()),
        ('classifier', MultiOutputClassifier(RandomForestClassifier()))
    ])
    
    # Do parameter tuning
    parameters = {
        #'classifier__estimator__n_estimators': [100, 200],
        #'classifier__estimator__criterion': ['gini', 'entropy'],
        'classifier__estimator__max_depth': [6, 9]
    }
    # Fit the model
    cv = GridSearchCV(pipeline, parameters, n_jobs=-1)
    
    return cv

def evaluate_model(model, X_test, Y_test, category_names):
    """
    Evaluate model performacne by printing out classification report.
    
    Args:
        model: fitted model
        X_test: test input of model
        Y_test: observed output
        category_names: for classification report
    Return:
        None
    """
    
    # Predict
    y_pred = model.predict(X_test)
    # Print classificaiton report
    for i in range(len(category_names)):
        print(category_names[i])
        print(classification_report(Y_test.iloc[:,i], y_pred[:,i]))

def save_model(model, model_filepath):
    """
    Save fitted model.
    Args:
        model: fitted model
        model_filepath: where to save the model
    """
    
    pickle_file = open(model_filepath, 'wb')
    pickle.dump(model, pickle_file)
    pickle_file.close()

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