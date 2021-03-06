import pandas as pd
import configargparse


def drop_file_rows(df, column_names, max_file_count):
    group_by_df = df.groupby(column_names).size().reset_index(name='count')
    rows_with_max_file_count = group_by_df[group_by_df['count'] > max_file_count]
    rows_with_max_file_count = rows_with_max_file_count.drop(['count'], axis=1)
    condition = (
            df['id'].isin(rows_with_max_file_count['id'])
            & df['version'].isin(rows_with_max_file_count['version'])
            & df['title'].isin(rows_with_max_file_count['title'])
            & df['description'].isin(rows_with_max_file_count['description'])
            & df['opendate'].isin(rows_with_max_file_count['opendate'])
            & df['fixdate'].isin(rows_with_max_file_count['fixdate'])
    )
    return df.drop(index=df[condition].index, axis=0)


def train_test_split(df, test_percentage=0.2):
    group_by_dataframe = df.groupby(['id', 'title', 'description', 'opendate']).size().reset_index(
        name='counts').sort_values(by='opendate', ascending=True, ignore_index=True)
    date_wise_count = group_by_dataframe['opendate'].to_frame().groupby(['opendate']).size().reset_index(name='count')
    date_wise_count['cumulative_count'] = date_wise_count['count'].cumsum()

    total = date_wise_count['cumulative_count'].iloc[-1]
    threshold = int(total * (1 - test_percentage))
    date_threshold = date_wise_count[date_wise_count['cumulative_count'] <= threshold].iloc[-1]['opendate']

    train = df[df['opendate'] <= date_threshold].sort_values(by='opendate', ascending=True, ignore_index=True)
    test = df[df['opendate'] > date_threshold].sort_values(by='opendate', ascending=True, ignore_index=True)

    return train, test


def split_dataset(max_file_count, test_percentage):
    dataset = pd.read_csv("Data/Java_Unified_Data.csv")
    dataset['opendate'] = pd.to_datetime(dataset['opendate'], format="%Y-%m-%d")
    dataset['fixdate'] = pd.to_datetime(dataset['fixdate'], format="%Y-%m-%d")
    column_names = ['id', 'version', 'title', 'description', 'opendate', 'fixdate']
    data_count = dataset.groupby(column_names).size().reset_index(name='counts')
    print("{} max file count will cover {}% data".format(max_file_count, round(
        (len(data_count[data_count['counts'] <= max_file_count]) / len(data_count)) * 100, 2)))
    dataset_after_drop = drop_file_rows(dataset, column_names, max_file_count)

    train_data, test_data = train_test_split(dataset_after_drop, test_percentage=test_percentage)
    train_data.fillna(value={"title": "", "description": ""}, inplace=True)
    test_data.fillna(value={"title": "", "description": ""}, inplace=True)
    train_data.to_csv("Data/Java_Train_Data.csv", index=False)
    test_data.to_csv("Data/Java_Test_Data.csv", index=False)


if __name__ == "__main__":
    p = configargparse.ArgParser()
    p.add('--max_file_count', required=True, is_config_file=False,
          help='Maximum allowed file count for per bug datapoint')
    p.add('--test_percentage', required=True, is_config_file=False, help='Test dataset percentage (such as 0.2)')
    options = p.parse_args()
    split_dataset(int(options.max_file_count), float(options.test_percentage))
