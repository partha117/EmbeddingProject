from pathlib import Path
import paramiko

def download_mrr_results():
    path = '/project/def-m2nagapp/partha9/Aster/'  # temporarily chdir to public
    for model in ["Extended_Roberta", "Reformer"]:
        for training in ["MLM", "QA", "Electra"]:
            for data in ["BLDS", "Bench-BLDS"]:
                for project in ["AspectJ", "Birt", "Eclipse_Platform_UI", "JDT", "SWT", "Tomcat"]:
                    temp_path = path + "Classifier_Text_" + model + "_" + training + "_" + data + "/" + "Result_" + project + ".txt"
                    save_path = "Results/" + (
                        model if model == "Reformer" else "Roberta") + "_" + training + "_" + data + "_" + project + ".json"
                    try:
                        client.get(temp_path, save_path)
                    except Exception:
                        print("Missing file {}".format(temp_path))
    client.close()

def download_rank_results():
    path = '/project/def-m2nagapp/partha9/Aster/'  # temporarily chdir to public
    for model in ["Extended_Roberta", "Reformer"]:
        for training in ["MLM", "QA", "Electra"]:
            for data in ["BLDS", "Bench-BLDS"]:
                    temp_path = path + "Classifier_Text_" + model + "_" + training + "_" + data + "/all_position_result.csv"
                    save_path = "Results/" + (
                        model if model == "Reformer" else "Roberta") + "_" + training + "_" + data + ".csv"
                    try:
                        client.get(temp_path, save_path)
                    except Exception as ex:
                        print(ex)
                        print("Missing file {}".format(temp_path))
    client.close()

def download_embedding_rank_results():
    path = '/project/def-m2nagapp/partha9/Aster/'  # temporarily chdir to public
    for model in ["Extended_Roberta", "Reformer"]:
        for training in ["MLM", "QA", "Electra"]:
                    temp_path = path + "Text_" + model + "_" + training + "/cosine_all_position_result.csv"
                    save_path = "Results/Embeddings/" + (
                        model if model == "Reformer" else "Roberta") + "_" + training + ".csv"
                    try:
                        client.get(temp_path, save_path)
                    except Exception:
                        print("Missing file {}".format(temp_path))
    client.close()
Path("Results/Embeddings/").mkdir(parents=True, exist_ok=True)
host = 'cedar.computecanada.ca'
username = 'partha9'
password = 'Chakraborty117@ca'
transport = paramiko.Transport(host)
transport.connect(username=username)
transport.auth_password(username, password)
transport.auth_interactive_dumb(username)
client = paramiko.SFTPClient.from_transport(transport)
#download_mrr_results()
download_rank_results()

