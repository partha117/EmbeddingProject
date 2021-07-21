import pandas as pd
import numpy as np


def calculate_mrr(load_file_name, save_file_name):
    df = pd.read_csv(load_file_name)
    project_list = sorted(df['project'].unique())
    model_list = sorted(list(set(df.columns.tolist()) - set(['BugId', 'CId', 'project'])))
    data = []
    for model in model_list:
        temp = dict()
        temp['overall'] = round(np.mean(1.0 / df[model]),3)
        for project in project_list:
            temp[project] = round(np.mean(1.0 / df[model][df['project'] == project]),3)
        data.append(temp)
    mrr_df = pd.DataFrame(data, index=model_list)
    mrr_df.to_csv(save_file_name)


calculate_mrr(load_file_name="Analyzed_Results/Embeddings_Rank_Analysis.csv", save_file_name="Analyzed_Results/Embeddings_Mrr.csv")
calculate_mrr(load_file_name="Analyzed_Results/Full_Models_Rank_Analysis.csv", save_file_name="Analyzed_Results/Full_Models_Mrr.csv")
