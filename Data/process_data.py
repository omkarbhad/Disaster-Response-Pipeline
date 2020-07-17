
def install_packages():

    """
    Installs the Required Packages, if they are not installed

    Arguments to Pass:
    ---- Nothing ----

    Returns:
    ---- Nothing ----
    """

    import sys
    import subprocess
    import pkg_resources

    required = {'sqlalchemy', 'pandas'}
    installed = {pkg.key for pkg in pkg_resources.working_set}
    missing = required - installed

    if missing:
        python = sys.executable
        subprocess.check_call([python, '-m', 'pip', 'install', *missing], stdout=subprocess.DEVNULL)


def get_data(messages_path, categories_path):
    """
    Get the Data of Messages and Categories from it's path and Merge it into a new Dataframe.

    Arguments to Pass:
    messages_path: String input -> Path containing messages.csv
    categories_path: String input -> Path containing categories.csv

    Returns:
    merged_df: Dataframe Output -> Merged Dataframe of messages.csv and categories.csv

    """
    import pandas as pd
    # Get the Datasets
    messages_df = pd.read_csv(messages_path)
    categories_df = pd.read_csv(categories_path)

    # Merge the Datasets
    merged_df = pd.merge(messages_df, categories_df, left_on='id', right_on='id', how='outer')

    return merged_df


def clean_data(merged_df):
    """
    Cleans the Merged Dataframe

    Arguments to Pass:
    merged_df : Dataframe input -> Merged Dataframe from Get Data Function

    Returns:
    cleaned_df : Dataframe Output -> Cleaned Dataframe

    """
    import pandas as pd
    # Split Categories column into Dataframe
    splitted_df = merged_df.categories.str.split(pat=';', expand=True)

    # Take the first row
    row = splitted_df.iloc[0]

    # Remove last two characters every string and get all the category names
    all_categories = row.apply(lambda k: k[:-2])

    # Set category names to Column names in the splitted dataframe
    splitted_df.columns = all_categories

    # Take the last character and convert to integer for all columns
    for category in all_categories:
        splitted_df[category] = splitted_df[category].str[-1].astype(int)

    # Drop the old categories column from Merged Dataframe
    merged_df.drop('categories', axis=1, inplace=True)

    # Concatenate the Old and Cleaned Dataframes
    cleaned_df = pd.concat([merged_df, splitted_df], axis=1, sort=False)

    # Remove the Duplicates
    cleaned_df.drop_duplicates(inplace=True)

    return cleaned_df


def save_data(cleaned_df, db_path):
    """
    Saves the Cleaned Dataframe to SQlite Database

    Arguments to Pass:
    cleaned_df : Dataframe input -> Cleaned Dataframe from Clean Data Function
    db_path : String input -> SQlite Database Database Path

    Returns:
    ---- Nothing ----

    """
    from sqlalchemy import create_engine
    engine = create_engine('sqlite:///{}'.format(db_path))
    cleaned_df.to_sql('Messages', engine, index=False, if_exists='replace')


def main():
    import sys

    if len(sys.argv) == 4:

        # Exclude the first Argument and Set Paths to respective variables
        messages_path, categories_path, db_path = sys.argv[1:]

        print('Getting data...\n    Messages Data: {}\n    Categories Data: {}'.format(messages_path, categories_path))
        merged_df = get_data(messages_path, categories_path)

        print('Cleaning data...')
        cleaned_df = clean_data(merged_df)

        print('Saving data...\n    Database: {}'.format(db_path))
        save_data(cleaned_df, db_path)

        print('Data is Cleaned and Saved to SQlite Database')


    else:
        print('Please Input Path of the messages.csv, categories.csv, Database Path '
              '\n\nExample: python process_data.py '
              'disaster_messages.csv disaster_categories.csv '
              'DisasterResponse.db')


if __name__ == '__main__':
    install_packages()
    main()
