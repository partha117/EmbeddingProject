import pandas as pd
from os import walk
import json

model_list = []
train_list = []
data_list = []
column_dict = {"Model Name": [], "Embedding Training Strategy": [], "Training Data (for CNN Model)": [], "AspectJ": [], "Birt": [], "Eclipse_Platform_UI": [], "JDT": [], "SWT": [], "Tomcat": []}
count = 1
decimal_point = 5
for file in sorted(next(walk("Results"), (None, None, []))[2]):
    try:
        performance = json.load(open("Results/{}".format(file),"r"))
        name_split = file.split("_")
        model = name_split[0]
        train = name_split[1]
        data = name_split[2]
        project = "_".join(name_split[3:]).strip()
        model = model if model == "Reformer" else "Long RoBERTa"
        train = "MLM and QA" if train == "QA" else train
        target_point = list(filter(lambda x: (x['Model Name'] == model) and (x["Embedding Training Strategy"] == train) and (x["Training Data (for CNN Model)"] == data), data_list))
        if len(target_point) <= 0:
            data_point = {
                'Model Name': model,
                "Embedding Training Strategy": train,
                "Training Data (for CNN Model)": data,
                project.split(".")[0]: round(performance['mean_reciprocal_rank'], decimal_point)

            }
            data_list.append(data_point)
        else:
            target_point[0][project.split(".")[0]] = round(performance['mean_reciprocal_rank'], decimal_point)
    except Exception:
        pass

df = pd.DataFrame(data_list)
df = df.sort_values(['Model Name', "Embedding Training Strategy","Training Data (for CNN Model)"], ascending = (True, True, True))
df.to_csv("Combined_Performance.csv", index=False)