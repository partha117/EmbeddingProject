import pandas as pd
import  numpy as np
df = pd.read_csv("./Embedding All .csv")
project_list = sorted(df['project'].unique())
model_list = sorted(list(set(df.columns) - set(['BugId','CId','project'])))
val = []
for model in model_list:
    temp_df = dict()
    temp_df['model_name'] = model
    for project in project_list:
        temp_df[project] = round(np.mean(1.0/df[df['project'] == project][model]),3)
    val.append(temp_df)
pd.DataFrame(val).to_csv("Only_Embedding_MRR.csv", index=False)