# import libraries
import sys
import pickle
import pandas as pd
import re
import nltk

nltk.download(['punkt', 'wordnet', 'averaged_perceptron_tagger'])

from sqlalchemy import create_engine
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from lightgbm import LGBMClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.multioutput import MultiOutputClassifier
from sklearn.ensemble import RandomForestClassifier

def load_data(database_filepath):
    """
    Loads the Data from SQLite Database

    Arguments to Pass:
    database_filepath : String input -> SQlite Database Database Path

    Returns:
    X: Message Column
    y: Target Categories
    category_name: list of all categories

    """

    engine = create_engine('sqlite:///{}'.format(database_filepath))
    saved_df = pd.read_sql_table(table_name='Messages', con=engine)
    X = saved_df.message.values

    y = saved_df.drop(['id', 'message', 'original', 'genre'], axis=1)
    y.loc[:, 'related'] = y['related'].replace(2, 1)

    category_name = y.columns

    return X, y, category_name


url_regex = r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'


def tokenize(text):
    """
    Processes the Text and Returns Cleaned Tokens

    Arguments to Pass:
    text: String input -> Raw Sentences

    Returns:
    clean_tokens: List output -> Contains tokenization of words from raw input sentences

    """
    detected_urls = re.findall(url_regex, text)
    for url in detected_urls:
        text = text.replace(url, "urlplaceholder")

    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()

    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)

    return clean_tokens


def build_model():
    """
    1. Machine Learning Pipeline which takes Message as input and Classifies Results.
    2. Uses CatBoost Algorithm to calculate the results with improved accuracy.

    Arguments to Pass:
    --- Nothing ---
    Returns:
    pipeline:  model output -> Model ready for prediction
    """
    pipeline = Pipeline([
        ('features', FeatureUnion([
            ('text_pipeline', Pipeline([
                ('vect', CountVectorizer(tokenizer=tokenize)),
                ('tfidf', TfidfTransformer())
            ]))
        ])),

        ('clf', MultiOutputClassifier(RandomForestClassifier()))
    ])

    #     parameters = {
    #         #'features__text_pipeline__vect__ngram_range': ((1, 1), (1, 2)),
    #         #'features__text_pipeline__vect__max_df': (0.5, 0.75, 1.0),
    #         #'features__text_pipeline__vect__max_features': (None, 5000, 10000),
    #         'features__text_pipeline__tfidf__use_idf': (True, False),
    #         #'clf__estimator__n_estimators': [50, 100, 200],
    #         'clf__estimator__min_samples_split': [2, 3, 4]
    #     }

    #     cv = GridSearchCV(pipeline, param_grid=parameters)

    return pipeline


def evaluate_model(model, X_test, Y_test, category_names):
    """

    Predict and Evaluate the Trained Model on the Test Data with F1-score and Other Parameters.

    Arguments to Pass:
    model: Machine Learning Model
    X_test: Messages to Test
    Y_test: Category values to Evaluate
    category_names: Category Names

    Return:
    --- Nothing ---
    """

    y_pred = model.predict(X_test)
    Y_pred = pd.DataFrame(data=y_pred,index=Y_test.index,columns=category_names)

    print(classification_report(Y_test, Y_pred, target_names=category_names))


def save_model(model, model_filepath):
    """

    Save the Trained Machine Learning Model

    Arguments to Pass:
    model: Machine Learning Model
    model_filepath: path where the model will be saved

    Returns:
    --- Nothing ---
    """
    pickle.dump(model, open(model_filepath, 'wb'))


def main():
    """
    Main Function that trains the Model and saves it.

    Arguments to Pass:
    arg1: Database Path
    arg2: Path where Models has to be saved

    Returns:
    --- Nothing ---
    """
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
        #save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '
              'as the first argument and the filepath of the pickle file to '
              'save the model to as the second argument. \n\nExample: python '
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()
