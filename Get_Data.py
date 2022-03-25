import pandas as pd
import uuid
import zlib


def file_converter(file_path):
    file = open(file_path, "r")
    return zlib.compress(file.read().encode("utf-8")).hex()


def get_embedding_dataset(file_path):
    print("collecting embedding data")
    df = pd.read_csv(file_path + "Data/Java_Unified_Data.csv")
    df['before_fix_uuid_file_path'] = df['before_fix_uuid_file_path'].map(lambda x: file_path + x)
    df['before_fix_uuid_file_path'] = df['before_fix_uuid_file_path'].map(lambda x: file_converter(x))
    column_names = ['id', 'report', 'before_fix_uuid_file_path', 'repository','version']
    accumulate_df = pd.DataFrame(columns=column_names)
    for row in df.sample(frac=1.0, random_state=13).reset_index(drop=True).iterrows():
        negative_sample = df[(df['id'] != row[1]['id']) & (df['title'] != row[1]['title']) & (
                df['github_repository'] == row[1]['github_repository'])].sample(frac=1, random_state=13)
        negative_sample['report'] = negative_sample['title'] + " " + negative_sample['description']
        drop_columns = set(negative_sample.columns.tolist()) - set(column_names)

        positive_sample = row[1].to_dict()
        positive_sample['report'] = positive_sample['title'] + " " + positive_sample['description']
        positive_sample = {key: value for key, value in positive_sample.items() if key in column_names}
        positive_sample['id'] = str(uuid.uuid4())
        positive_sample['match'] = 1

        negative_sample.drop(columns=drop_columns, inplace=True)
        negative_sample['id'] = positive_sample['id']
        negative_sample['match'] = 0

        accumulate_df = accumulate_df.append(positive_sample, ignore_index=True)
        accumulate_df = accumulate_df.append(negative_sample, ignore_index=True)

    accumulate_df.reset_index(drop=True, inplace=True)
    accumulate_df.rename(columns={'id': 'cid', 'before_fix_uuid_file_path': 'file_content'}, inplace=True)
    return accumulate_df

if __name__ == "__main__":
    file_path = "/scratch/partha9/"
    df = get_embedding_dataset(file_path)
    df.to_csv("BLDS_With_Project.csv", index=False)