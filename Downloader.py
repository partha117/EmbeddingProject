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
                    except Exception:
                        print("Missing file {}".format(temp_path))
    client.close()
Path("Results").mkdir(parents=True, exist_ok=True)
host = 'cedar.computecanada.ca'
username = 'partha9'
password = 'Chakraborty117@ca'
transport = paramiko.Transport(host)
transport.connect(username=username)
transport.auth_password(username, password)
transport.auth_interactive_dumb(username)
client = paramiko.SFTPClient.from_transport(transport)
download_rank_results()

