import transformers
from tree_sitter import Language, Parser
from pathlib import Path
from tokenizers import ByteLevelBPETokenizer
import pandas as pd
from tokenizers.processors import BertProcessing, RobertaProcessing
from tree_sitter import Language, Parser
import os
from pathlib import Path
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from transformers import DataCollatorForLanguageModeling
from transformers import Trainer, TrainingArguments
from transformers import AdamW, RobertaModel, AutoModel, RobertaTokenizer, AutoModelForMaskedLM, RobertaForMaskedLM, \
    RobertaForSequenceClassification, RobertaConfig, AutoConfig, AutoModelForSequenceClassification, AutoTokenizer
import zlib
import pickle
import numpy as np
import torch.nn as nn
import torch.optim as optim
from datetime import datetime
import gc
from tqdm.notebook import tqdm as ntqdm
from tqdm import tqdm
import sys
sys.path.append(os.path.abspath("/home/partha9/EmbeddingProject"))
from Reformer import ElectraUtil
from Reformer.ElectraUtil import ElectraForPreTraining
import argparse
import uuid


def freeze_model(model, model_name):
    for param in getattr(model, model_name).parameters() if model_name else model.parameters():
        param.requires_grad = False
    return model


def count_parameters(m: torch.nn.Module, only_trainable: bool = False):
    """
    returns the total number of parameters used by `m` (only counting
    shared parameters once); if `only_trainable` is True, then only
    includes parameters with `requires_grad = True`
    """
    parameters = m.parameters()
    if only_trainable:
        parameters = list(p for p in parameters if p.requires_grad)
    unique = dict((p.data_ptr(), p) for p in parameters).values()
    return sum(p.numel() for p in unique)


def get_combined_full_dataset(project_name, test_percentage=0.4):
    df1 = pd.read_csv("{}.csv".format(project_name), delimiter='\t')
    df2 = pd.read_csv("{}_features.csv".format(project_name))
    df3 = pd.read_csv("{}_features_file_content.csv".format(project_name))
    df4 = pd.merge(df2, df3, left_on='cid', right_on='cid', how='inner')
    df5 = pd.merge(df1, df4, left_on='id', right_on='report_id', how='inner')
    df5['report'] = df5['summary'] + df5['description']
    df5['project_name'] = project_name.split("/")[-1]
    train_pos, test_pos = train_test_split(df5[df5['match'] == 1], test_size=test_percentage, random_state=13, shuffle=False)
    train, test = df5[df5['bug_id'].isin(train_pos['bug_id'])], df5[df5['bug_id'].isin(test_pos['bug_id'])]
    test.to_csv("/project/def-m2nagapp/partha9/Aster/" + "{}_test.csv".format(project_name.split("/")[-1]), index=False)
    train = train.copy().reset_index(drop=True)
    small_train = pd.DataFrame(columns=train.columns)
    for item in train['bug_id'].unique():
        temp = pd.concat((train[(train['bug_id'] == item) & (train['match'] == 1)],
                          train[(train['bug_id'] == item) & (train['match'] == 0)].head(10)))
        small_train = pd.concat((small_train, temp))
    small_train.drop(columns=set(small_train.columns) - {'id', 'cid', 'report', 'file_content', 'match'}, inplace=True)
    return small_train


def create_random_dataset(dataset_list, full_size=1000):
    part_size = 1.0 / len(dataset_list)
    temp_df = pd.DataFrame(columns=dataset_list[0].columns)
    for item in dataset_list:
        temp_df = temp_df.append(
            item.sample(frac=(full_size * part_size) / len(item), replace=False, random_state=13).reset_index(
                drop=True)).reset_index(drop=True)
    return temp_df.sample(frac=1, random_state=13).reset_index(drop=True)


class ClassificationHead(nn.Module):
    def __init__(self, config, embed_dim, kernel_num=3, kernel_sizes=[2, 3, 4, 5], num_labels=2):
        super().__init__()
        self.num_labels = num_labels
        self.embed_dim = embed_dim
        self.kernel_num = kernel_num
        self.kernel_sizes = kernel_sizes
        self.conv_layers = nn.ModuleList(
            [nn.Conv2d(1, self.kernel_num, (k, self.embed_dim)) for k in self.kernel_sizes])
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.out_proj = nn.Linear(len(self.kernel_sizes) * self.kernel_num, self.num_labels)

    def forward(self, hidden_states, **kwargs):
        output = hidden_states.unsqueeze(1)
        output = [nn.functional.relu(conv(output)).squeeze(3) for conv in self.conv_layers]
        output = [nn.functional.max_pool1d(i, i.size(2)).squeeze(2) for i in output]
        output = torch.cat(output, 1)
        output = self.dropout(output)
        logits = self.out_proj(output)
        return logits


class ElectraClassification(nn.Module):

    def __init__(self, num_labels, base_model, config, kernel_num=3, kernel_sizes=[2, 3, 4, 5]):
        super(ElectraClassification, self).__init__()
        self.num_labels = num_labels
        self.config = config
        self.transformer = base_model
        self.classifier = ClassificationHead(config=config, embed_dim=config.max_position_embeddings if isinstance(base_model, transformers.ReformerModel) else config.hidden_size, kernel_num=kernel_num,
                                             kernel_sizes=kernel_sizes, num_labels=num_labels)

    def forward(self, input_ids):
        output = self.transformer(input_ids)
        output = self.classifier(output[0])
        return output


def file_converter(file_path):
    file = open(file_path, "r")
    return zlib.compress(file.read().encode("utf-8")).hex()


def get_embedding_dataset(file_path):
    df = pd.read_csv(file_path + "Data/Java_Unified_Data.csv")
    df['before_fix_uuid_file_path'] = df['before_fix_uuid_file_path'].map(lambda x: file_path + x)
    df['before_fix_uuid_file_path'] = df['before_fix_uuid_file_path'].map(lambda x: file_converter(x))
    column_names = ['id', 'report', 'before_fix_uuid_file_path']
    accumulate_df = pd.DataFrame(columns=column_names)
    for row in df.sample(frac=340 / len(df), random_state=13).reset_index(drop=True).iterrows():
        negative_sample = df[(df['id'] != row[1]['id']) & (df['title'] != row[1]['title']) & (
                df['github_repository'] == row[1]['github_repository'])].sample(frac=1, random_state=13).head(14)
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
        self.dataset.drop(columns=['report', 'file_content'], inplace=True)
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
        codefile = open(self.tmp + str(features[self.map['cid']]) + "_content.txt", "r")
        codefile_content = codefile.read()

        reportfile = open(self.tmp + str(features[self.map['cid']]) + "_report.txt", "r")
        reportfile_content = reportfile.read()
        code_data = zlib.decompress(bytes.fromhex(codefile_content)).decode()
        combined_data = reportfile_content + " " + code_data
        # return features[self.map['cid']], self.tokenizer.encode_plus(combined_data,truncation=True, max_length=self.max_size)['input_ids'], features[self.map['match']]
        return features[self.map['cid']], combined_data, features[self.map['match']]


def batch_parser(content_list, report_list=None):
    temp = []
    JAVA_LANGUAGE = Language('build/my-languages.so', 'java')
    parser = Parser()
    parser.set_language(JAVA_LANGUAGE)
    for i, item in enumerate(content_list):
        temp.append(report_list[i] + " " + parser.parse(bytes(item, 'utf-8')).root_node.sexp())
    return temp


def get_label_weight(all_labels):
    unique_labels, label_count = np.unique(all_labels, return_counts=True)
    total_count = np.sum(label_count)
    label_weight = total_count / label_count
    return [label_weight[int(item)] for item in all_labels]


if __name__ == "__main__":
    # root_path = "/project/def-m2nagapp/partha9/Aster/PlainRobertaWithAst_Size_Extension_Classifier"
    # Path(root_path).mkdir(parents=True, exist_ok=True)
    #
    # model_path = "/project/def-m2nagapp/partha9/Aster/PlainRobertaWithAst_Size_Extension/train_output/checkpoint-18000/"
    # project_name = "/project/def-m2nagapp/partha9/Dataset/CombinedData/"
    # scratch_path = "/scratch/partha9/Dataset/"
    # tokenizer_root = '/project/def-m2nagapp/partha9/Aster/PlainRobertaWithAst_Size_Extension'
    # token_max_size = 1498
    # batch_size = 64
    # model_name = "roberta"
    sys.modules['ElectraUtil'] = ElectraUtil
    parser = argparse.ArgumentParser()
    parser.add_argument("--root_path", default=None, type=str, help="")
    parser.add_argument("--model_path", default=None, type=str, help="")
    parser.add_argument("--project_name", default=None, type=str, help="")
    parser.add_argument("--scratch_path", default=None, type=str, help="")
    parser.add_argument("--checkpoint", default=None, type=str, help="")
    parser.add_argument("--tokenizer_root", default=None, type=str, help="")
    parser.add_argument("--token_max_size", default=None, type=int, help="")
    parser.add_argument("--batch_size", default=None, type=int, help="")
    parser.add_argument("--model_name", default=None, type=str, help="")
    parser.add_argument('--combined_data', action='store_true', help="")
    parser.add_argument('--embedding_data', action='store_true', help="")
    parser.add_argument('--electra', action='store_true', help="")
    parser.add_argument('--state_dict', action='store_true', help="")
    args = parser.parse_args()
    args.root_path += "_BLDS" if args.embedding_data else "_Bench-BLDS"
    Path(args.root_path).mkdir(parents=True, exist_ok=True)
    file_path = "/scratch/partha9/"
    dev = "cuda:0" if torch.cuda.is_available() else "cpu"
    tokenizer = RobertaTokenizer(args.tokenizer_root + "/tokenizer/aster-vocab.json",
                                 args.tokenizer_root + "/tokenizer/aster-merges.txt")

    if args.combined_data:
        df1, df2, df3, df4, df5, df6 = get_combined_full_dataset(
            "/project/def-m2nagapp/partha9/Dataset/Birt"), get_combined_full_dataset(
            "/project/def-m2nagapp/partha9/Dataset/AspectJ"), get_combined_full_dataset(
            "/project/def-m2nagapp/partha9/Dataset/Tomcat"), get_combined_full_dataset(
            "/project/def-m2nagapp/partha9/Dataset/SWT"), get_combined_full_dataset(
            "/project/def-m2nagapp/partha9/Dataset/JDT"), get_combined_full_dataset(
            "/project/def-m2nagapp/partha9/Dataset/Eclipse_Platform_UI")
        combined_df = create_random_dataset([df1, df2, df3, df4, df5, df6], full_size=5000)
    elif args.embedding_data:
        combined_df = get_embedding_dataset(file_path=file_path)
    dataset = BugDataset(project_name=args.project_name, scratch_path=args.scratch_path, dataframe=combined_df,parser=parser)
    # config = AutoConfig.from_pretrained(args.model_path,
    #                                     num_labels=1)  # RobertaConfig.from_pretrained(model_path, num_labels=1)

    if args.electra:
        if args.state_dict:
            temp_config = AutoConfig.from_pretrained(args.dis_model_name_or_path)
            temp_config.is_decoder = False
            temp_config.output_hidden_states = True
            full_base_model_dict = torch.load(args.model_path + args.checkpoint)
            full_base_model = ElectraForPreTraining(temp_config)
            full_base_model.load_state_dict(full_base_model_dict)
            model = freeze_model(full_base_model.electra, args.model_name)
        else:
            full_base_model = torch.load(args.model_path + args.checkpoint)
            model = freeze_model(full_base_model, args.model_name)
        model = ElectraClassification(num_labels=1, base_model=model,
                                  config=full_base_model.electra.config, kernel_num=3, kernel_sizes=[2, 3, 4, 5])
    else:
        full_base_model = AutoModel.from_pretrained(args.model_path + args.checkpoint)
        model = freeze_model(full_base_model, args.model_name)
        model = ElectraClassification(num_labels=1, base_model=model,
                                      config=full_base_model.config, kernel_num=3, kernel_sizes=[2, 3, 4, 5])
    model.to(dev)
    Path(args.root_path + "_Dataset/{}/".format(args.project_name.split("/")[-2])).mkdir(parents=True, exist_ok=True)
    pickle.dump(dataset, open(
        args.root_path + "_Dataset/{}/{}_full_dataset.pickle".format(args.project_name.split("/")[-2],
                                                                     args.project_name.split("/")[-2]),
        "wb"))



    unique_labels = np.unique(dataset.get_all_label())
    label_weight = get_label_weight(dataset.get_all_label())
    sampler = WeightedRandomSampler(label_weight, len(label_weight), replacement=True)

    dataloader = DataLoader(dataset, batch_size=int(args.batch_size), pin_memory=True, num_workers=2, sampler=sampler,
                            drop_last=True)
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    loss_list = []

    Path(args.root_path + "_Model").mkdir(parents=True, exist_ok=True)
    print("Starting Epoch")
    exit(0)
    for epoch in range(1, 7):  # loop over the dataset multiple times

        epoch_loss = []
        epoch_start_time = datetime.now()
        loop = tqdm(dataloader, leave=True)
        print("Starting Loop")
        for i, data in enumerate(loop):
            # print("Here1")
            iter_start_time = datetime.now()
            _, combined_input, labels = data
            # print(code)
            # print(labels)
            # code_ast_tree = parser.parse(bytes(code, 'utf-8')).root_node.sexp()
            # combined_input = report + " " + code_ast_tree

            combined_input, labels = \
                tokenizer.batch_encode_plus(combined_input, truncation=True, max_length=args.token_max_size,
                                            padding=True,
                                            return_tensors='pt')[
                    'input_ids'], torch.tensor(labels, dtype=torch.float64).to(dev)
            # print("Here2")
            # zero the parameter gradients
            optimizer.zero_grad()
            outputs = model(input_ids=combined_input.to(dev))
            loss = criterion(torch.sigmoid(outputs.view(-1).double()).to(dev), labels.double().to(dev))
            loss_list.append(loss.item())
            epoch_loss.append(loss)
            loss.backward()
            optimizer.step()
            gc.collect()
            # print("Here4")
            loop.set_description('Epoch {}'.format(epoch))
            loop.set_postfix(loss=round(loss.item(), 4), duration=(datetime.now() - iter_start_time).seconds)
        torch.save(model, args.root_path + "_Electra_Model/Model_{}".format(epoch + 1))
        print("------------------------{} Epoch Completed----------------".format(epoch))
        epoch_loss = sum(epoch_loss) / len(epoch_loss)
        print("--------------Epoch Loss {} Time Elpased: {}---------------".format(epoch_loss, (
                datetime.now() - epoch_start_time).seconds))

    print('Finished Training')
