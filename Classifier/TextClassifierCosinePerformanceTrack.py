import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
import zlib
import os
import sys
from scipy.spatial.distance import cdist
sys.path.append(os.path.abspath("/home/partha9/EmbeddingProject"))
from Reformer import ElectraUtil
from Reformer.ElectraUtil import ElectraForPreTraining
from Classifier.TextTransformerClassifier import ElectraClassification, ClassificationHead, get_combined_full_dataset, \
    batch_parser, freeze_model
from pathlib import Path
import numpy as np
import pickle
from transformers import AutoTokenizer, AutoModel, RobertaTokenizer
import json
from sklearn.metrics import roc_auc_score, average_precision_score
from tqdm import tqdm
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel, AutoConfig
import argparse
from tree_sitter import Language, Parser


class BugDataset(Dataset):

    def __init__(self, project_name, dataframe, scratch_path=None, parser=None):
        self.tmp = scratch_path
        self.tmp += project_name.split("/")[-1] + "/"
        # JAVA_LANGUAGE = Language('build/my-languages.so', 'java')
        # self.parser = Parser()
        # self.parser.set_language(JAVA_LANGUAGE)
        self.dataset = dataframe.copy()
        if not os.path.isdir(self.tmp):
            Path(self.tmp).mkdir(parents=True, exist_ok=True)
        for item in self.dataset.iterrows():
            # print(item[1])
            if not os.path.isfile(self.tmp + str(item[1]['cid']) + "_report.txt"):
                file = open(self.tmp + str(item[1]['cid']) + "_report.txt", "w")
                file.write(str(item[1]['report']))
                file.close()
            if not os.path.isfile(self.tmp + str(item[1]['cid']) + "_content.txt"):
                file = open(self.tmp + str(item[1]['cid']) + "_content.txt", "w")
                file.write(item[1]['file_content'])
                file.close()
        # self.dataset.drop(columns=['report', 'file_content'], inplace=True)
        self.map = {name: index for index, name in enumerate(self.dataset.columns.tolist())}
        self.dataset = self.dataset.to_numpy()
        self.parser = parser

    def __len__(self):
        return len(self.dataset)

    def get_all_label(self):
        return self.dataset[:, self.map['match']]

    def get_match_value(self, indices):
        return self.dataset[indices, self.map['match']]

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        features = self.dataset[idx]
        reportfile_content = features[self.map['report']]
        code_data = zlib.decompress(bytes.fromhex(features[self.map['file_content']])).decode()
        return features[self.map['cid']], str(reportfile_content), code_data, features[self.map['match']]


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
    position_array = []
    if torch.cuda.is_available():
        dev = "cuda:0"
    else:
        dev = "cpu"

    JAVA_LANGUAGE = Language('/home/partha9/build/my-languages.so', 'java')
    parser = Parser()
    parser.set_language(JAVA_LANGUAGE)
    positive_test_data = positive_test_data if len(positive_test_data) <= 100 else positive_test_data.sample(
        frac=100.0 / len(positive_test_data), random_state=13)
    print("Data Length: {}".format(len(positive_test_data)))

    loop_tqdm = tqdm(range(len(positive_test_data)), leave=True)
    for i in loop_tqdm:
        print((i / len(positive_test_data)) * 100)
        # ToDo: Verify K
        test_dataset = get_sample_set(idx=i, positive_test_data=positive_test_data,
                                      combined_full_dataset=combined_full_dataset, k=30)
        test_bugdataset = BugDataset(project_name=project_name, dataframe=test_dataset, scratch_path=scratch_path,
                                     parser=parser)
        test_dataloader = DataLoader(test_bugdataset, batch_size=30, pin_memory=True, num_workers=1)
        # data = next(iter(test_dataloader))
        all_code_embedding = None
        all_report_embedding = None
        true_positive_location = 0
        for batch in test_dataloader:
            _, report_input, code_input, labels = batch

            report_input, code_input = \
                tokenizer.batch_encode_plus(report_input, truncation=True, max_length=token_max_size, padding=True,
                                            return_tensors='pt')[
                    'input_ids'], \
                tokenizer.batch_encode_plus(code_input, truncation=True, max_length=token_max_size, padding=True,
                                            return_tensors='pt')[
                    'input_ids']
            model.eval()
            print("Labels", labels)
            with torch.no_grad():
                code_embedding = model(input_ids=code_input.to(dev))[0].mean(1).detach().cpu().numpy()
                report_embedding = model(input_ids=report_input.to(dev))[0].mean(1).detach().cpu().numpy()
                if all_code_embedding is None and all_report_embedding is None:
                    all_code_embedding = code_embedding
                    all_report_embedding = report_embedding
                else:
                    all_code_embedding = all_code_embedding.append(code_embedding)
                    all_report_embedding = all_report_embedding.append(report_embedding)
            true_positive_location += np.where(labels == 1)[0] if np.any(labels.numpy() == 1) else len(labels)
        # mrr calculation

        print("Code Shape", all_code_embedding.shape)
        print("Report Shape", all_report_embedding.shape)
        similarity = 1 - cdist(all_code_embedding, all_report_embedding, metric='cosine')
        true_positive_value = similarity[0][true_positive_location]
        rank = np.max(np.sum(similarity > true_positive_value, axis=-1) + 1)

        print("Rank", rank)
        position_array.append({
            "BugId": positive_test_data.iloc[i]["bug_id"],
            "position": rank
        })

    return pd.DataFrame(position_array)


if __name__ == "__main__":
    sys.modules['ElectraUtil'] = ElectraUtil
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default=None, type=str, help="")
    parser.add_argument('--scratch_path', help='Model Number')
    parser.add_argument('--batch_size', help='Model Number')
    parser.add_argument('--model_path', help='Model Number')
    parser.add_argument('--tokenizer_path', help='Model Number')
    parser.add_argument('--test_data_path', help='Model Number')
    parser.add_argument("--model_name", default=None, type=str, help="")
    parser.add_argument('--token_max_size', help='Model Number')
    parser.add_argument('--electra', action='store_true', help="")
    parser.add_argument('--state_dict', action='store_true', help="")
    parser.add_argument('--config', default=None, type=str, help="")
    options = parser.parse_args()

    scratch_path = options.scratch_path
    model_path = options.model_path
    tokenizer_path = options.tokenizer_path
    test_data_path = options.test_data_path

    token_max_size = int(options.token_max_size)
    batch_size = int(options.batch_size)
    dev = "cuda:0" if torch.cuda.is_available() else "cpu"
    if options.electra:
        if options.state_dict:
            temp_config = AutoConfig.from_pretrained(options.config)
            temp_config.is_decoder = False
            temp_config.output_hidden_states = True
            full_base_model_dict = torch.load(options.model_path + options.checkpoint)
            full_base_model = ElectraForPreTraining(temp_config)
            full_base_model.load_state_dict(full_base_model_dict)
            model = freeze_model(full_base_model.electra, options.model_name)
        else:
            full_base_model = torch.load(options.model_path + options.checkpoint)
            model = freeze_model(full_base_model.electra, options.model_name)
    else:
        full_base_model = AutoModel.from_pretrained(options.model_path + options.checkpoint)
        model = freeze_model(full_base_model, options.model_name)
    model.to(dev)
    tokenizer = RobertaTokenizer(tokenizer_path + "tokenizer/aster-vocab.json",
                                 tokenizer_path + "tokenizer/aster-merges.txt")

    project_list = ["AspectJ", "Birt", "Tomcat", "SWT", "JDT", "Eclipse_Platform_UI"]
    all_result = None
    for project_name in project_list:
        dump_file = open(model_path + "Result_{}.txt".format(project_name), "w")
        print("Starting Project {}".format(project_name))
        test_dataset = pd.read_csv(test_data_path + "{}_test.csv".format(project_name))
        positive_test_data = get_positive_dataset(test_dataset)
        dataframe = calculate_metrices(test_dataset,
                                       positive_test_data,
                                       project_name, tokenizer,
                                       scratch_path=scratch_path)

        dataframe['project'] = project_name
        if all_result is None:
            all_result = dataframe
        else:
            all_result = pd.concat([all_result, dataframe])

    all_result.to_csv(model_path + "cosine_all_position_result.csv", index=False)
