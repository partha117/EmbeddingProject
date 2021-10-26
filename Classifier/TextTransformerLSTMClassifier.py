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
from torch.optim.lr_scheduler import ReduceLROnPlateau

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
    df5.to_csv("{}_complete.csv".format(project_name), index=False)
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

def get_splitted_tensor(tensor, max_size, overlap_size):
    return tensor.unfold(dimension=1,size=max_size, step= max_size - overlap_size)

def reshape_input(tensor):
    num_sentences, max_segments, segment_length = tensor.size()
    total_segments = num_sentences * max_segments
    print(tensor.shape)
    print(total_segments, segment_length)
    tensor_ = tensor.reshape(total_segments, segment_length)
    return tensor_, num_sentences, max_segments

def get_last_layer_output(tensor, num_sentences, max_segments ):
    idxs_sentences = torch.arange(num_sentences)
    idx_last_output = max_segments - 1
    return tensor[idxs_sentences, idx_last_output]

class ClassificationHead(nn.Module):
    def __init__(self, input_size, embed_size, num_labels=2):
        super().__init__()
        self.input_size = input_size
        self.num_labels = num_labels
        self.embed_size = embed_size
        self.code_lstm = nn.LSTM(self.input_size, self.embed_size, 1, bias=True, batch_first=True)
        self.report_lstm = nn.LSTM(self.input_size, self.embed_size, 1, bias=True, batch_first=True)
        self.dropout = nn.Dropout(0.15)
        self.out_proj = nn.Linear(self.embed_size * 2, self.num_labels)

    def forward(self, code_hidden_states, report_hidden_states, code_properties, report_properties):
        code_lstm_output, _ = self.code_lstm(code_hidden_states)
        report_lstm_output, _ = self.report_lstm(report_hidden_states)
        code_last_layer = get_last_layer_output(self.dropout(code_lstm_output), code_properties['num_sentences'], code_properties['max_segments'])
        report_last_layer = get_last_layer_output(self.dropout(report_lstm_output), report_properties['num_sentences'], report_properties['max_segments'])
        combined_input = torch.cat([code_last_layer, report_last_layer], dim=1)
        logits = self.out_proj(combined_input)
        return logits


class ClassifierModel(nn.Module):

    def __init__(self, config, num_labels, base_model, embed_size,
                 code_overlap_size=0, report_overlap_size=0):
        super(ClassifierModel, self).__init__()
        self.num_labels = num_labels
        self.config = config
        self.transformer = base_model
        self.code_max_size = 2048 if isinstance(base_model, transformers.ReformerModel) else config.hidden_size
        self.report_max_size = 2048 if isinstance(base_model, transformers.ReformerModel) else config.hidden_size
        self.code_overlap_size = code_overlap_size
        self.report_overlap_size = report_overlap_size
        self.dropout = nn.Dropout(0.2)
        self.classifier = ClassificationHead(
            input_size=self.code_max_size,
            embed_size=embed_size, num_labels=num_labels)

    def forward(self, code_input_id, report_input_id, code_properties, report_properties):
        # c_in = get_splitted_tensor(code_input_id, max_size=self.code_max_size, overlap_size=self.code_overlap_size)
        # r_in = get_splitted_tensor(report_input_id, max_size=self.report_max_size,
        #                            overlap_size=self.report_overlap_size)
        #
        # c_in_, c_num_sentences, c_max_segments = reshape_input(c_in)
        # r_in_, r_num_sentences, r_max_segments = reshape_input(r_in)

        code_output = self.dropout(self.transformer(code_input_id)[1])
        report_output = self.dropout(self.transformer(report_input_id)[1])

        converted_c_out = code_output.view(code_properties['num_sentences'], code_properties['max_segments'], code_output.shape[-1])
        converted_r_out = report_output.view(report_properties['num_sentences'], report_properties['max_segments'], report_output.shape[-1])
        output = self.classifier(converted_c_out, converted_r_out,
                                 code_properties=code_properties,
                                 report_properties=report_properties)
        return output


def file_converter(file_path):
    file = open(file_path, "r")
    return zlib.compress(file.read().encode("utf-8")).hex()


def get_embedding_dataset(file_path):
    print("collecting embedding data")
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
        #self.dataset.drop(columns=['report', 'file_content'], inplace=True)
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
        # combined_data = str(reportfile_content) + " " + code_data
        return features[self.map['cid']], str(reportfile_content), code_data, features[self.map['match']]


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
    parser.add_argument("--model_no", default=None, type=str, help="")
    parser.add_argument("--lr_rate", default=str(0.002), type=str, help="")
    parser.add_argument('--combined_data', action='store_true', help="")
    parser.add_argument('--embedding_data', action='store_true', help="")
    parser.add_argument('--electra', action='store_true', help="")
    parser.add_argument('--state_dict', action='store_true', help="")
    parser.add_argument('--pretrained', action='store_true', help="")
    parser.add_argument('--config', default=None, type=str, help="")
    parser.add_argument('--embed_size', default=str(256), type=str, help="")
    parser.add_argument('--overlap_size', default=str(0), type=str, help="")
    args = parser.parse_args()
    args.root_path += "_BLDS" if args.embedding_data else "_Bench-BLDS"
    Path(args.root_path).mkdir(parents=True, exist_ok=True)
    file_path = "/scratch/partha9/"
    dev = "cuda:0" if torch.cuda.is_available() else "cpu"
    tokenizer = RobertaTokenizer(args.tokenizer_root + "/tokenizer/aster-vocab.json",
                                 args.tokenizer_root + "/tokenizer/aster-merges.txt")

    if args.combined_data:
        print("collecting combined data")
        df1, df2, df3, df4, df5, df6 = get_combined_full_dataset(
            "/project/def-m2nagapp/partha9/Dataset/Birt"), get_combined_full_dataset(
            "/project/def-m2nagapp/partha9/Dataset/AspectJ"), get_combined_full_dataset(
            "/project/def-m2nagapp/partha9/Dataset/Tomcat"), get_combined_full_dataset(
            "/project/def-m2nagapp/partha9/Dataset/SWT"), get_combined_full_dataset(
            "/project/def-m2nagapp/partha9/Dataset/JDT"), get_combined_full_dataset(
            "/project/def-m2nagapp/partha9/Dataset/Eclipse_Platform_UI")
        combined_df = create_random_dataset([df1, df2, df3, df4, df5, df6], full_size=5000)
        combined_df.to_csv("Bench_BLDS_Dataset.csv", index=False)
    elif args.embedding_data:
        combined_df = get_embedding_dataset(file_path=file_path)
        combined_df.to_csv("BLDS_Dataset.csv", index=False)
    dataset = BugDataset(project_name=args.project_name, scratch_path=args.scratch_path, dataframe=combined_df,parser=None)
    # config = AutoConfig.from_pretrained(args.model_path,
    #                                     num_labels=1)  # RobertaConfig.from_pretrained(model_path, num_labels=1)
    print("Loading models")
    if args.pretrained:
        model = torch.load(args.root_path + "/Model_{}".format(args.model_no))
    else:
        if args.electra:
            if args.state_dict:
                temp_config = AutoConfig.from_pretrained(args.config)
                temp_config.is_decoder = False
                temp_config.output_hidden_states = True
                full_base_model_dict = torch.load(args.model_path + args.checkpoint)
                full_base_model = ElectraForPreTraining(temp_config)
                full_base_model.load_state_dict(full_base_model_dict)
                model = freeze_model(full_base_model.electra, args.model_name)
            else:
                full_base_model = torch.load(args.model_path + args.checkpoint)
                model = freeze_model(full_base_model.electra, args.model_name)
            model = ClassifierModel(num_labels=1, base_model=model,
                                    config=full_base_model.electra.config, embed_size=int(args.embed_size), code_overlap_size=args.overlap_size, report_overlap_size=args.overlap_size)
        else:
            full_base_model = AutoModel.from_pretrained("roberta-base")
            model = freeze_model(full_base_model, args.model_name)
            # ToDo: Pass only the model
            # ToDo: Edit sh file
            model = ClassifierModel(num_labels=1, base_model=model,
                                    config=full_base_model.config, embed_size=int(args.embed_size), code_overlap_size = int(args.overlap_size), report_overlap_size=int(args.overlap_size))
    model.to(dev)



    unique_labels = np.unique(dataset.get_all_label())
    label_weight = get_label_weight(dataset.get_all_label())
    sampler = WeightedRandomSampler(label_weight, len(label_weight), replacement=True)

    dataloader = DataLoader(dataset, batch_size=int(args.batch_size), pin_memory=True, num_workers=2, sampler=sampler,
                            drop_last=True)
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=float(args.lr_rate))
    scheduler = ReduceLROnPlateau(optimizer, 'min', patience=40, cooldown=20, verbose=True)
    loss_list = []

    Path(args.root_path + "_Model").mkdir(parents=True, exist_ok=True)
    print("Starting Epoch")
    for epoch in range(1, 10):  # loop over the dataset multiple times

        epoch_loss = []
        epoch_start_time = datetime.now()
        loop = tqdm(dataloader, leave=True)
        print("Starting Loop")
        for i, data in enumerate(loop):
            # print("Here1")
            iter_start_time = datetime.now()
            _, report, code, labels = data
            # print(code)
            # print(labels)
            # code_ast_tree = parser.parse(bytes(code, 'utf-8')).root_node.sexp()
            # combined_input = report + " " + code_ast_tree

            report_input, code_input, labels = \
                tokenizer.batch_encode_plus(report,max_length=16 * args.token_max_size, pad_to_multiple_of=args.token_max_size,
                                            truncation=True,
                                            padding=True,
                                            return_tensors='pt')[
                    'input_ids'], \
                tokenizer.batch_encode_plus(code,max_length=16 * args.token_max_size, pad_to_multiple_of=args.token_max_size,
                                            truncation=True,
                                            padding=True,
                                            return_tensors='pt')[
                    'input_ids'], torch.tensor(labels, dtype=torch.float64).to(dev)
            # print("Here2")
            # zero the parameter gradients
            print("Code shape", code_input.shape)
            report_input, code_input = report_input.to(dev), code_input.to(dev)
            optimizer.zero_grad()
            c_in = get_splitted_tensor(code_input, max_size=int(args.embed_size),
                                       overlap_size=int(args.overlap_size))
            r_in = get_splitted_tensor(report_input, max_size=int(args.embed_size),
                                       overlap_size=int(args.overlap_size))

            c_in_, c_num_sentences, c_max_segments = reshape_input(c_in)
            r_in_, r_num_sentences, r_max_segments = reshape_input(r_in)
            outputs = model(code_input_id=c_in_.to(dev), report_input_id=r_in_.to(dev),code_properties={"num_sentences": c_num_sentences, "max_segments": c_max_segments}, report_properties={"num_sentences": r_num_sentences, "max_segments": r_max_segments})
            loss = criterion(torch.sigmoid(outputs.view(-1).double()).to(dev), labels.double().to(dev))
            loss_list.append(loss.item())
            epoch_loss.append(loss)
            loss.backward()
            # optimizer.step()
            scheduler.step(loss)
            gc.collect()
            # print("Here4")
            loop.set_description('Epoch {}'.format(epoch))
            loop.set_postfix(loss=round(loss.item(), 4), duration=(datetime.now() - iter_start_time).seconds)
        torch.save(model, args.root_path + "/Model_LSTM_{}_Embed_size_{}_Overlap_size_{}".format(epoch, args.embed_size, args.overlap_size))
        torch.save(model.state_dict(), args.root_path + "/Model_LSTM_State_Dict_{}_Embed_size_{}_Overlap_size_{}".format(epoch, args.embed_size,
                                                                                                 args.overlap_size))
        print("------------------------{} Epoch Completed----------------".format(epoch))
        epoch_loss = sum(epoch_loss) / len(epoch_loss)
        print("--------------Epoch Loss {} Time Elpased: {}---------------".format(epoch_loss, (
                datetime.now() - epoch_start_time).seconds))

    print('Finished Training')
