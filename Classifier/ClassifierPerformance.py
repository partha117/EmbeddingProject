import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
import zlib
import os
import sys
sys.path.append(os.path.abspath("/home/partha9/EmbeddingProject"))
from Classifier.TransformerClassifierElectra import ElectraClassification, ClassificationHead, BugDataset, get_combined_full_dataset, batch_parser
from pathlib import Path
import numpy as np
import pickle
from transformers import AutoTokenizer, AutoModel, RobertaTokenizer
import json
from sklearn.metrics import roc_auc_score, average_precision_score
from tqdm import tqdm
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel
import argparse
from tree_sitter import Language, Parser


def get_positive_dataset(test_dataset):
    return test_dataset[test_dataset['match'] == 1].reset_index(drop=True)


def get_sample_set(idx, positive_test_data, combined_full_dataset, k=30,
                   features=['id', 'cid', 'report', 'file_content', 'match']):
    positive_set = positive_test_data.iloc[idx][features]
    negative_set = combined_full_dataset[(combined_full_dataset['bug_id'] == positive_test_data.iloc[idx].bug_id) &
                                         (combined_full_dataset['match'] == 0)].sort_values('rVSM_similarity',
                                                                                            ascending=False).head(
        k).reset_index(drop=True)[features]
    return negative_set.append(positive_set).sample(frac=1.0, random_state=13).reset_index(drop=True)


def calculate_metrices(combined_full_dataset, positive_test_data, project_name, tokenizer, strict=False,
                       scratch_path=None):
    top_k_counter = [0] * 20
    mrr_value = np.array([])
    map_value = np.array([])
    if torch.cuda.is_available():
        dev = "cuda:0"
    else:
        dev = "cpu"
    global_y_true = np.array([])
    global_y_predicted = np.array([])

    JAVA_LANGUAGE = Language('build/my-languages.so', 'java')
    parser = Parser()
    parser.set_language(JAVA_LANGUAGE)
    positive_test_data = positive_test_data if len(positive_test_data) <= 100 else positive_test_data.sample(frac=100.0/len(positive_test_data),random_state=13)
    print("Data Length: {}".format(len(positive_test_data)))

    loop_tqdm = tqdm(range(len(positive_test_data)), leave=True)
    for i in loop_tqdm:
        # print((i / len(positive_test_data)) * 100)
        #ToDo: Verify K
        test_dataset = get_sample_set(idx=i, positive_test_data=positive_test_data,
                                      combined_full_dataset=combined_full_dataset, k=10)
        test_bugdataset = BugDataset(project_name=project_name, dataframe=test_dataset, scratch_path=scratch_path)
        test_dataloader = DataLoader(test_bugdataset, batch_size=30, pin_memory=True, num_workers=1)
        # data = next(iter(test_dataloader))

        y_true = np.array([])
        y_predicted = np.array([])
        for batch in test_dataloader:
            _,  report_inputs, code_inputs, labels = batch

            # code_ast_tree = parser.parse(bytes(code_inputs, 'utf-8')).root_node.sexp()
            combined_input = batch_parser(code_inputs,report_inputs)

            combined_input, labels = \
                tokenizer.batch_encode_plus(combined_input, truncation=True, max_length=token_max_size, padding=True,
                                            return_tensors='pt')[
                    'input_ids'], torch.tensor(labels, dtype=torch.float64).to(dev)
            model.eval()
            with torch.no_grad():
                eval_result = model(input_ids=combined_input.to(dev))
                eval_result = torch.sigmoid(eval_result.logits.view(-1).double())
            #print(eval_result)
            y_true = np.append(y_true, labels.cpu().numpy().ravel())
            y_predicted = np.append(y_predicted, eval_result.cpu().numpy().ravel())

        # mrr calculation
        sorted_prediction_rank = np.argsort(-y_predicted)
        sorted_prediction_value = np.array([y_true[item] for item in sorted_prediction_rank])
        # print(sorted_prediction_value)
        lowest_retrieval_rank = (sorted_prediction_value == 0).argsort(axis=0)
        # print(lowest_retrieval_rank, lowest_retrieval_rank[0])
        mrr_value = np.append(mrr_value, np.array(1.0 / (lowest_retrieval_rank[0] + 1)))

        # average precision calculation
        map_value = np.append(map_value, np.array(average_precision_score(y_true, y_predicted)))

        # Top K calculation
        sorted_label_rank = np.argsort(-y_true)
        for position_k in range(0, 20):
            common_item = np.intersect1d(sorted_prediction_rank[:(position_k + 1)],
                                         sorted_label_rank[:(position_k + 1)])
            if len(common_item) > 0:
                top_k_counter[position_k] += 1

        global_y_true = np.append(global_y_true, y_true)
        global_y_predicted = np.append(global_y_predicted, y_predicted)

    # calculation of map
    mean_average_precision = np.mean(map_value)

    # calculation of auc
    auc_score = roc_auc_score(global_y_true, global_y_predicted)

    # calculation of mrr
    mean_reciprocal_rank = np.mean(mrr_value)

    # calculation of top k
    acc_dict = {}
    for i, counter in enumerate(top_k_counter):
        acc = counter / (len(positive_test_data))
        acc_dict[i + 1] = round(acc, 3)

    return acc_dict, mean_reciprocal_rank, mean_average_precision, auc_score


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_no', help='Model Number')
    parser.add_argument('--scratch_path', help='Model Number')
    parser.add_argument('--batch_size', help='Model Number')
    parser.add_argument('--model_path', help='Model Number')
    parser.add_argument('--tokenizer_path', help='Model Number')
    parser.add_argument('--test_data_path', help='Model Number')
    parser.add_argument('--token_max_size', help='Model Number')
    options = parser.parse_args()


    model_no = int(options.model_no)
    scratch_path = options.scratch_path
    model_path = options.model_path
    tokenizer_path = options.tokenizer_path
    test_data_path = options.test_data_path


    token_max_size = int(options.token_max_size)
    batch_size = int(options.batch_size)
    dev = "cuda:0" if torch.cuda.is_available() else "cpu"
    model = torch.load(model_path + "Model_{}".format(model_no))
    tokenizer = RobertaTokenizer(tokenizer_path + "tokenizer/aster-vocab.json",
                                 tokenizer_path + "tokenizer/aster-merges.txt")

    project_list = ["AspectJ", "Birt", "Tomcat","SWT","JDT","Eclipse_Platform_UI"]
    for project_name in project_list:
        dump_file = open(model_path + "Result_{}.txt".format(project_name), "w")
        print("Starting Project {}".format(project_name))
        test_dataset = pd.read_csv(test_data_path + "{}_test.csv".format(project_name))
        positive_test_data = get_positive_dataset(test_dataset)
        acc_dict, mean_reciprocal_rank, mean_average_precision, auc_score = calculate_metrices(test_dataset,
                                                                                               positive_test_data,
                                                                                               project_name, tokenizer,
                                                                                               scratch_path=scratch_path)

        acc_dict['mean_reciprocal_rank'] = mean_reciprocal_rank
        acc_dict['mean_average_precision'] = mean_average_precision
        acc_dict['auc_score'] = auc_score
        json.dump(acc_dict, dump_file)
