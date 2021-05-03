from pathlib import Path
from tokenizers import ByteLevelBPETokenizer
import pandas as pd
from tokenizers.processors import BertProcessing, RobertaProcessing
from tree_sitter import Language, Parser
import os
from pathlib import Path
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import DataCollatorForLanguageModeling, RobertaConfig, ReformerConfig, XLNetConfig, XLMConfig, \
    XLMRobertaConfig
from transformers import Trainer, TrainingArguments
from transformers import AdamW, RobertaModel, AutoModel, RobertaTokenizer, AutoModelForMaskedLM, RobertaForMaskedLM
from joblib import Parallel, delayed
from tree_sitter import Language, Parser
import re


def build_lib():
    Language.build_library(
        # Store the library in the `build` directory
        '/home/partha9/build/my-languages.so',

        # Include one or more languages
        [
            'tree-sitter-java',
        ]
    )


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


def convert_file_to_ast(file_path, parser):
    file = open(file_path, "r")
    file_content = file.read()
    tree = parser.parse(bytes(file_content, "utf-8"))
    return tree.root_node.sexp()


def save_file(path, item):
    JAVA_LANGUAGE = Language('/project/def-m2nagapp/partha9/Data/build/my-languages.so', 'java')
    parser = Parser()
    parser.set_language(JAVA_LANGUAGE)
    before_fix_uuid_name = item[1]['before_fix_uuid_file_path'].split("/")[-1].split(".")[0]
    before_fix_file = open(path + "Data/AST_Files/{}.txt".format(before_fix_uuid_name), "w")
    before_fix_file.write(convert_file_to_ast(path + item[1]['before_fix_uuid_file_path'], parser))
    before_fix_file.close()

    after_fix_uuid_name = item[1]['after_fix_uuid_file_path'].split("/")[-1].split(".")[0]
    after_fix_file = open(path + "Data/AST_Files/{}.txt".format(after_fix_uuid_name), "w")
    after_fix_file.write(convert_file_to_ast(path + item[1]['after_fix_uuid_file_path'], parser))
    after_fix_file.close()


def create_ast_files():
    if not os.path.isdir(scratch_path + "partha9/Data/AST_Files/"):
        Path(scratch_path + "partha9/Data/AST_Files/").mkdir(parents=True, exist_ok=True)
        df = pd.read_csv(scratch_path + "partha9/Data/Java_Unified_Data_with_SHA.csv")
        path = scratch_path + "partha9/"
        Parallel(n_jobs=-1)(  # Uses all cores but one
            delayed(save_file)(path, item)
            for item in df.iterrows()
        )


def file_reader(ast_paths, report_paths):
    if not isinstance(ast_paths, str):
        accumulate = []
        for ast, report in zip(ast_paths, report_paths):
            temp = ""
            with open(report, "r") as file:
                temp += file.read()
            with open(ast, "r") as file:
                temp += file.read()
            accumulate.append(temp)
    else:
        acuumulate = None
        temp = ""
        with open(report_paths, "r") as file:
            temp += file.read()
        with open(ast_paths, "r") as file:
            temp += file.read()
        accumulate = temp
    return accumulate


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
            before_fix_ast_path = scratch_path + "partha9/Data/AST_Files/" + get_uuid(
                rows['before_fix_uuid_file_path']) + ".txt"
            report_files = scratch_path + "partha9/Data/Report_Files/" + get_uuid(
                rows['before_fix_uuid_file_path']) + ".txt"
        else:
            before_fix_ast_path = rows['before_fix_uuid_file_path'].map(
                lambda x: scratch_path + "partha9/Data/AST_Files/" + get_uuid(x) + ".txt").tolist()
            report_files = rows['before_fix_uuid_file_path'].map(
                lambda x: scratch_path + "partha9/Data/Report_Files/" + get_uuid(x) + ".txt").tolist()
        temp = file_reader(before_fix_ast_path, report_files)
        return self.tokenizer.encode_plus(temp, truncation=True, max_length=512)['input_ids']


if __name__ == "__main__":
    scratch_path = "/scratch/"
    root_path = "/project/def-m2nagapp/partha9/Aster/PlainRobertaWithAst_Size_Extension/"
    Path(root_path).mkdir(parents=True, exist_ok=True)
    train_data, val_data = train_test_split(pd.read_csv(scratch_path + "partha9/Data/Java_Train_Data.csv"),
                                            test_size=0.125)
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    create_java_only_dataset()
    create_report_files()
    create_ast_files()
    before_fix_ast_paths = train_data['before_fix_uuid_file_path'].map(
        lambda x: scratch_path + "partha9/Data/AST_Files/" + get_uuid(x) + ".txt").tolist()
    after_fix_ast_paths = train_data['after_fix_uuid_file_path'].map(
        lambda x: scratch_path + "partha9/Data/AST_Files/" + get_uuid(x) + ".txt").tolist()
    report_files = train_data['before_fix_uuid_file_path'].map(
        lambda x: scratch_path + "partha9/Data/Report_Files/" + get_uuid(x) + ".txt").tolist()
    all_file_path = before_fix_ast_paths + report_files
    if not os.path.isfile(root_path + "/tokenizer/aster-vocab.json"):
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
    temp_dataset = BugDataset(scratch_path + "partha9/Data/Java_Train_Data.csv")
    temp_dataloader = DataLoader(temp_dataset, batch_size=4, num_workers=1)
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer, mlm=True, mlm_probability=0.15
    )
    Path(root_path + "/train_output/").mkdir(parents=True, exist_ok=True)
    training_args = TrainingArguments(
        output_dir=root_path + "/train_output/",
        overwrite_output_dir=True,
        num_train_epochs=4,
        per_device_train_batch_size=2,
        save_steps=500,
        save_total_limit=4,
        dataloader_drop_last=True

    )
    model = RobertaForMaskedLM.from_pretrained('roberta-base')
    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=BugDataset(dataframe=train_data, tokenizer=tokenizer),
        #     eval_dataset= BugDataset(dataframe=val_data,tokenizer=tokenizer)
    )
    trainer.train()
