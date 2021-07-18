import pandas as pd
from os import walk
import json

def json_formatter():
    model_list = []
    train_list = []
    data_list = []
    column_dict = {"Model Name": [], "Embedding Training Strategy": [], "Training Data (for CNN Model)": [],
                   "AspectJ": [], "Birt": [], "Eclipse_Platform_UI": [], "JDT": [], "SWT": [], "Tomcat": []}
    count = 1
    decimal_point = 2
    for file in sorted(next(walk("Results"), (None, None, []))[2]):
        try:
            performance = json.load(open("Results/{}".format(file), "r"))
            name_split = file.split("_")
            model = name_split[0]
            train = name_split[1]
            data = name_split[2]
            project = "_".join(name_split[3:]).strip()
            model = model if model == "Reformer" else "Long RoBERTa"
            train = "MLM and QA" if train == "QA" else train
            target_point = list(filter(
                lambda x: (x['Model Name'] == model) and (x["Embedding Training Strategy"] == train) and (
                            x["Training Data (for CNN Model)"] == data), data_list))
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
    df = df.sort_values(['Model Name', "Embedding Training Strategy", "Training Data (for CNN Model)"],
                        ascending=(True, True, True))
    df.to_csv("Combined_Performance.csv", index=False)

def csv_formatter():
    all_ranks =  None
    for model in ["Extended_Roberta", "Reformer"]:
        for training in ["MLM", "QA", "Electra"]:
            for data in ["BLDS", "Bench-BLDS"]:
                name = (
                        model if model == "Reformer" else "Roberta") + "_" + training + "_" + data

                path = "Results/" + name + ".csv"
                df = pd.read_csv(path)
                df.sort_values(by=['BugId', 'CId', 'project'], inplace=True, ignore_index=True)
                if all_ranks is None:
                    df.rename(columns={"position": name}, inplace=True)
                    all_ranks = df
                else:
                    all_ranks[name] = df['position']
    all_ranks.to_csv("Full_Models_Rank_Analysis.csv", index=False)

def csv_formatter_embeddings():
    all_ranks =  None
    for model in ["Extended_Roberta", "Reformer"]:
        for training in ["MLM", "QA", "Electra"]:
                name = (
                        model if model == "Reformer" else "Roberta") + "_" + training

                path = "Results/Embeddings/" + name + ".csv"
                df = pd.read_csv(path)
                df.sort_values(by=['BugId', 'CId', 'project'], inplace=True, ignore_index=True)
                if all_ranks is None:
                    df.rename(columns={"position": name}, inplace=True)
                    all_ranks = df
                else:
                    all_ranks[name] = df['position']
    all_ranks.to_csv("Embeddings_Rank_Analysis.csv", index=False)
def get_matrix_corr(model=True):
    if model:
        df = pd.read_csv("Full_Models_Rank_Analysis.csv")
    else:
        df = pd.read_csv("Embeddings_Rank_Analysis.csv")
    df.drop(columns=['BugId', 'CId', 'project'], inplace=True)
    mat_corr = df.corr()
    if model:
        mat_corr.to_csv("Full_Models_Correlation.csv")
    else:
        mat_corr.to_csv("Embeddings_Correlation.csv")
csv_formatter_embeddings()
get_matrix_corr(model=False)