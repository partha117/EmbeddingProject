from transformers.deepspeed import  HfDeepSpeedConfig
from pathlib import Path
import torch.distributed as dist
from tokenizers import ByteLevelBPETokenizer
import pandas as pd
from math import ceil, floor
from tokenizers.processors import BertProcessing, RobertaProcessing
import os
from pathlib import Path
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import DataCollatorForLanguageModeling, RobertaConfig, ReformerConfig, XLNetConfig, XLMConfig, \
    XLMRobertaConfig, deepspeed
from transformers import Trainer, TrainingArguments
from transformers import AdamW, RobertaModel, AutoModel, RobertaTokenizer, AutoModelForMaskedLM, RobertaForMaskedLM, \
    RobertaForQuestionAnswering
from joblib import Parallel, delayed
from torch.optim._multi_tensor import AdamW as DistributedAdamW
import numpy as np
from torch.utils.data import DataLoader
from transformers import AdamW
from tqdm import tqdm
import re
import deepspeed
import fairscale
from datetime import timedelta

def create_java_only_dataset():
    if not os.path.isfile(scratch_path + "partha9/Data/Java_Unified_Data_with_SHA.csv"):
        df = pd.read_csv("Data/Unified_Data_with_SHA.csv")
        df2 = df[df["language_name"] == 'Java']
        df2.reset_index(drop=True, inplace=True)
        df2.to_csv("Data/Java_Unified_Data_with_SHA.csv", index=False)


def get_uuid(text):
    return text.split("/")[-1].split(".")[0]


def remove_comments_and_docstrings(source):
    def replacer(match):
        s = match.group(0)
        if s.startswith('/'):
            return " "  # note: a space and not an empty string
        else:
            return s

    pattern = re.compile(
        r'//.*?$|/\*.*?\*/|\'(?:\\.|[^\\\'])*\'|"(?:\\.|[^\\"])*"',
        re.DOTALL | re.MULTILINE
    )
    temp = []
    for x in re.sub(pattern, replacer, source).split('\n'):
        if x.strip() != "":
            temp.append(x)
    return '\n'.join(temp)


def create_report_files():
    if not os.path.isdir(scratch_path + "partha9/Data/Report_Files/"):
        Path(scratch_path + "partha9/Data/Report_Files/").mkdir(parents=True, exist_ok=True)
        df = pd.read_csv(scratch_path + "partha9/Data/Java_Unified_Data_with_SHA.csv")
        for item in df.iterrows():
            uuid_name = item[1]['before_fix_uuid_file_path'].split("/")[-1].split(".")[0]
            file = open(scratch_path + "partha9/Data/Report_Files/{}.txt".format(uuid_name), "w")
            file.write(item[1]['title'] + " " + item[1]['description'])
            file.close()


def file_reader(before_fix_ast_paths, after_fix_ast_path, report_paths):
    if not isinstance(before_fix_ast_paths, str):
        accumulate = [[], [], []]
        for before_ast, after_ast, report in zip(before_fix_ast_paths, after_fix_ast_path, report_paths):
            with open(report, "r") as file:
                accumulate[0].append(file.read())
            with open(before_ast, "r") as file:
                accumulate[1].append(file.read())
            with open(after_ast, "r") as file:
                accumulate[2].append(file.read())
    else:
        accumulate = []
        with open(report_paths, "r") as file:
            accumulate.append(file.read())
        with open(before_fix_ast_paths, "r") as file:
            accumulate.append(file.read())
        with open(after_fix_ast_path, "r") as file:
            accumulate.append(file.read())
    return accumulate


def find_difference(before, after):
    before, after = np.array(before), np.array(after)
    maxlength = max(len(before), len(after))
    padded_before = before if len(before) == maxlength else np.pad(before, (0, maxlength - len(before)),
                                                                   constant_values=-1)
    padded_after = after if len(after) == maxlength else np.pad(after, (0, maxlength - len(after)), constant_values=-1)
    difference = np.where(padded_before != padded_after)
    # print("-----------------------------------------------------\n")
    # print(difference)
    if len(difference[0]) == 0:
        return torch.tensor([0]), torch.tensor([0])
    start = difference[0][0]
    end = len(before) - 1 if len(before) < maxlength else difference[0][-1]
    return torch.tensor([start]), torch.tensor([end])


class BugDataset(Dataset):

    def __init__(self, file_path=None, dataframe=None, tokenizer=None):
        if file_path is not None:
            self.dataset = pd.read_csv(file_path)
        else:
            self.dataset = dataframe
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        rows = self.dataset.iloc[idx, :]
        if isinstance(idx, int):
            before_fix_file_path = scratch_path + "partha9/Data/UUID_Files/" + \
                                   rows['before_fix_uuid_file_path'].split("/")[-1]
            after_fix_file_path = scratch_path + "partha9/Data/UUID_Files/" + \
                                  rows['after_fix_uuid_file_path'].split("/")[-1]
            report_files = scratch_path + "partha9/Data/Report_Files/" + get_uuid(
                rows['before_fix_uuid_file_path']) + ".txt"
        else:
            before_fix_file_path = rows['before_fix_uuid_file_path'].map(
                lambda x: scratch_path + "partha9/Data/UUID_Files/" + x.split("/")[-1]).tolist()
            after_fix_file_path = rows['after_fix_uuid_file_path'].map(
                lambda x: scratch_path + "partha9/Data/UUID_Files/" + x.split("/")[-1]).tolist()
            report_files = rows['before_fix_uuid_file_path'].map(
                lambda x: scratch_path + "partha9/Data/Report_Files/" + get_uuid(x) + ".txt").tolist()
        temp = file_reader(before_fix_file_path, after_fix_file_path, report_files)
        before, after = self.tokenizer.encode_plus(temp[0], temp[1], truncation=True, max_length=1498)['input_ids'], \
                        self.tokenizer.encode_plus(temp[0], temp[2], truncation=True, max_length=1498)['input_ids']
        start, end = find_difference(before, after)
        report_context = self.tokenizer.encode_plus(temp[0], temp[1], truncation=True, max_length=1498, padding=True,
                                                    pad_to_multiple_of=1498)
        return {'input_ids': torch.tensor(report_context['input_ids']),
                'attention_mask': torch.tensor(report_context['attention_mask']), 'start_positions': start,
                'end_positions': end}


if __name__ == "__main__":
    scratch_path = "/scratch/"
    root_path = "/project/def-m2nagapp/partha9/Aster/Text_Extended_Roberta_QA/"
    Path(root_path).mkdir(parents=True, exist_ok=True)
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    create_java_only_dataset()
    create_report_files()
    train_data, val_data = train_test_split(pd.read_csv(scratch_path + "partha9/Data/Java_Train_Data.csv"),
                                            test_size=0.125)
    before_fix_file_paths = train_data['before_fix_uuid_file_path'].map(
        lambda x: scratch_path + "partha9/Data/UUID_Files/" + x.split("/")[-1]).tolist()
    after_fix_file_paths = train_data['after_fix_uuid_file_path'].map(
        lambda x: scratch_path + "partha9/Data/UUID_Files/" + x.split("/")[-1]).tolist()
    report_files = train_data['before_fix_uuid_file_path'].map(
        lambda x: scratch_path + "partha9/Data/Report_Files/" + get_uuid(x) + ".txt").tolist()
    all_file_path = before_fix_file_paths + report_files
    if not os.path.isfile(root_path + "/tokenizer/aster-vocab.json"):
        print("starting tokenizer train")
        tokenizer = ByteLevelBPETokenizer()
        tokenizer.train(files=all_file_path, min_frequency=2, special_tokens=[
            "<s>",
            "<pad>",
            "</s>",
            "<unk>",
            "<mask>",
        ])
        Path(root_path + "/tokenizer/").mkdir(parents=True, exist_ok=True)
        tokenizer.save_model(root_path + "/tokenizer/", "./aster")
    tokenizer = RobertaTokenizer(root_path + "/tokenizer/aster-vocab.json", root_path + "/tokenizer/aster-merges.txt")
    dist.init_process_group(backend='nccl', init_method="env://", rank=0, world_size=2, timeout=timedelta(minutes=5))
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    save_at = 500
    model = RobertaForQuestionAnswering.from_pretrained(
        "/project/def-m2nagapp/partha9/Aster/Text_Extended_Roberta_MLM" + "/train_output/" + "checkpoint-25000/")
    train_dataset = BugDataset(dataframe=train_data, tokenizer=tokenizer)
    model.to(device)
    total_layers = sum(1 for item in model.parameters())
    model = fairscale.nn.Pipe(model, balance=[ceil(total_layers * 0.5), floor(total_layers * 0.5)], devices=[0, 1], chunks=4)
    # optim = AdamW(model.parameters(), lr=5e-5)
    optim = DistributedAdamW(model.parameters(), lr=5e-5)
    train_loader = DataLoader(train_dataset, batch_size=10, shuffle=True)

    for epoch in range(3):
        # set model to train mode
        model.train()
        # setup loop (we use tqdm for the progress bar)
        loop = tqdm(train_loader, leave=True)
        for i, batch in enumerate(loop):
            # initialize calculated gradients (from prev step)
            optim.zero_grad()
            # pull all the tensor batches required for training
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            start_positions = batch['start_positions'].to(device)
            end_positions = batch['end_positions'].to(device)
            # train model on batch and return outputs (incl. loss)
            outputs = model(input_ids, attention_mask=attention_mask,
                            start_positions=start_positions,
                            end_positions=end_positions)
            # extract loss
            loss = outputs[0]
            # calculate loss for every parameter that needs grad update
            loss.backward()
            # update parameters
            optim.step()
            # print relevant info to progress bar
            loop.set_description("Epoch {}".format(epoch))
            loop.set_postfix(loss=loss.item())
            if i % save_at == 0:
                model.save_pretrained(
                    root_path + "/train_output/" + "CheckPoint-{}".format(
                        i))
