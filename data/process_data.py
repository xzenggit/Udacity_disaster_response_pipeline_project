import sys
import pandas as pd
from sqlalchemy import create_engine

def load_data(messages_filepath, categories_filepath):
    """
    Load message and categories files and merge them.
    
    Args:
        messages_filepath: path of message file
        categories_filepath: path of categroy file
    Return:
        df: merged dataframe
    """
    
    # Load data files
    messages = pd.read_csv(messages_filepath, index_col=0)
    categories = pd.read_csv(categories_filepath, index_col=0)
    # Merge datasets
    df = messages.merge(categories, on='id')

    return df

def clean_data(df):
    """
    Clean data, convert text to categories.
    Args:
        df: output from load_data() function.
    Return:
        df: clean verison of data.
    """
    # Get column names of categories
    categories = df['categories'].str.split(';', expand=True)
    row = categories.iloc[0,]
    category_colnames = [x[:-2] for x in row]
    # Rename the columns of `categories`
    categories.columns = category_colnames
    
    for column in categories:
        # set each value to be the last character of the string
        categories[column] = categories[column].str[-1]
        # convert column from string to numeric
        categories[column] = categories[column].astype(int)
        
    # Replace categories column in df with categroies dataframe
    df = df.drop(['categories'], axis=1)
    df = df.merge(categories, on='id')
    # drop duplicates
    df = df.drop_duplicates()
    
    return df

def save_data(df, database_filename):
    """"
    Save data into database.
    Args:
        df: output of clean_data() function
        database_filename: name of database file
    Return:
        None
    """
    
    engine = create_engine('sqlite:///'+database_filename)
    df.to_sql('DisasterResponse', engine, index=False) 


def main():
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        df = load_data(messages_filepath, categories_filepath)

        print('Cleaning data...')
        df = clean_data(df)
        
        print('Saving data...\n    DATABASE: {}'.format(database_filepath))
        save_data(df, database_filepath)
        
        print('Cleaned data saved to database!')
    
    else:
        print('Please provide the filepaths of the messages and categories '\
              'datasets as the first and second argument respectively, as '\
              'well as the filepath of the database to save the cleaned data '\
              'to as the third argument. \n\nExample: python process_data.py '\
              'disaster_messages.csv disaster_categories.csv '\
              'DisasterResponse.db')


if __name__ == '__main__':
    main()