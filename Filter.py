import pandas as pd
import numpy as np

def variable_filter(series, threshold, greater_or_equal=False):
    return series >= threshold if greater_or_equal else series < threshold


def filter_file(df, threshold, greater_or_equal=False):
    model_names = list(set(df.columns) - set(["BugId","CId","project"]))
    condition = None
    for item in model_names:
        if condition is None:
            condition = variable_filter(df[item], threshold=threshold, greater_or_equal=greater_or_equal)
        else:
            condition = condition & variable_filter(df[item], threshold=threshold,greater_or_equal=greater_or_equal)
    return df[condition].reset_index(drop=True).sort_values(by=['project','BugId', 'CId'])

if __name__ =="__main__":
    threshold = 5
    df = pd.read_csv("Analyzed_Results/Embeddings_Rank_Analysis.csv")
    filtered_df_less = filter_file(df, threshold=threshold, greater_or_equal=False)
    filtered_df_greater_or_equal = filter_file(df, threshold=threshold, greater_or_equal=True)
    filtered_df_less.to_csv("Analyzed_Results/Embeddings_All_Good.csv",index=False)
    filtered_df_greater_or_equal.to_csv("Analyzed_Results/Embeddings_All_Bad.csv", index=False)

    df = pd.read_csv("Analyzed_Results/Full_Models_Rank_Analysis.csv")
    filtered_df_less = filter_file(df, threshold=threshold, greater_or_equal=False)
    filtered_df_greater_or_equal = filter_file(df, threshold=threshold, greater_or_equal=True)
    filtered_df_less.to_csv("Analyzed_Results/Full_Models_All_Good.csv", index=False)
    filtered_df_greater_or_equal.to_csv("Analyzed_Results/Full_Models_All_Bad.csv", index=False)
