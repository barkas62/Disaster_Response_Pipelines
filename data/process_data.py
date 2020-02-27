import sys
import os
import pandas as pd
from sqlalchemy import create_engine

def load_data(messages_filepath, categories_filepath):
    '''
    Loads the message and category data from csv files
    :param messages_filepath  : string: Messages csv filepath
    :param categories_filepath: string: Categories csv filepath
    :return: DataFrame, containing combined messages and categories data
    '''

    # read from csv files
    messages_df   = pd.read_csv(messages_filepath)
    categories_df = pd.read_csv(categories_filepath)

    # merge datasets
    df = pd.merge(messages_df, categories_df.drop_duplicates(subset=['id']), how='left')

    # create a dataframe of the 36 individual category columns
    categories = categories_df['categories'].str.split(';', expand=True)

    # select the first row of the categories dataframe
    row = categories.iloc[0]

    # use this row to extract a list of new column names for categories.
    # one way is to apply a lambda function that takes everything
    # up to the second to last character of each string with slicing
    category_colnames = [colname[:-2] for colname in row.tolist()]

    # rename the columns of `categories`
    categories.columns = category_colnames

    # Convert category values to just numbers 0 or 1
    for column in categories:
        # set each value to be the last character of the string, converted to int
        categories[column] = categories[column].apply(lambda val: int(val[-1]))

    # drop the original categories column from `df`
    df.drop('categories', axis=1, inplace=True)

    # concatenate the original DataFrame with the new `categories` DataFrame
    df = pd.concat([df, categories], axis=1)

    return df


def clean_data(df):
    '''
    Cleans the data: drops rows with NaN values and duplicate rows
    :param df: DataFrame : Input DataFrame
    :return: cleaned dataframe
    '''
    # drop rows with NaN values in 'message' and all categories columns
    important_columns = ['message'] + df.columns[4:].tolist()
    df.dropna(subset=important_columns, inplace=True)

    # Find duplicate rows (with same message values)
    duplicated = df.duplicated(subset='message')

    # drop duplicates and return the resulting dataframe
    df = df[~duplicated]

    # there is a small number of rows with 'related' values equal 2; drop them
    drop_index = df[df['related'] == 2].index
    df = df.drop(drop_index)

    return df


def save_data(df, database_filename):
    '''
    Saves a DataFrame to a sqlite db
    :param df: DataFrame : Input DataFrame
    :param database_filename: Output sqlite db filename
    :return: None
    '''
    engine = create_engine('sqlite:///'+database_filename)
    df.to_sql('labeled_messages', engine, index=False, if_exists='replace')
    engine.dispose()


def main():
    if len(sys.argv) == 4:
        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        if not os.path.isfile(messages_filepath):
            print("Messages file {} does not exist. Please provide a valid filepath.".format(messages_filepath))
            return

        if not os.path.isfile(categories_filepath):
            print("Categories file {} does not exist. Please provide a valid filepath.".format(categories_filepath))
            return

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))

        df = load_data(messages_filepath, categories_filepath)

        print('Cleaning data...')
        df = clean_data(df)

        print('Cleaned data contain '+ str(df.shape[0]) + " rows")
        
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