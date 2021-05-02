#!/usr/bin/env python
# coding: utf-8

# In[1]:


# ! git clone https://github.com/tree-sitter/tree-sitter-java
# ! cp /project/def-m2nagapp/partha9/Dataset/my-languages.so /scratch-deleted-2021-mar-20/partha9/


# In[2]:


# from tree_sitter import Language, Parser

# Language.build_library(
#   # Store the library in the `build` directory
#   '/home/partha9/build/my-languages.so',

#   # Include one or more languages
#   [
#     'tree-sitter-java',
#   ]
# )


# In[3]:


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
from transformers import AdamW, RobertaModel, AutoModel, RobertaTokenizer, AutoModelForMaskedLM, RobertaForMaskedLM, \
    RobertaForQuestionAnswering
from joblib import Parallel, delayed
import numpy as np
from torch.utils.data import DataLoader
from transformers import AdamW
from tqdm import tqdm

# In[4]:

scratch_path = "/scratch/"
root_path = "/project/def-m2nagapp/partha9/Aster/PlainRobertaWithAst_Size_Extension_QA/"
Path(root_path).mkdir(parents=True, exist_ok=True)

# In[5]:


# !mkdir /scratch-deleted-2021-mar-20/partha9/Data/
# !unzip /project/def-m2nagapp/partha9/Data/graham_upload.zip  -d /scratch-deleted-2021-mar-20/partha9/Data/


# In[6]:


device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')


# In[7]:


def create_java_only_dataset():
    if not os.path.isfile(scratch_path + "partha9/Data/Java_Unified_Data_with_SHA.csv"):
        df = pd.read_csv("Data/Unified_Data_with_SHA.csv")
        df2 = df[df["language_name"] == 'Java']
        df2.reset_index(drop=True, inplace=True)
        df2.to_csv("Data/Java_Unified_Data_with_SHA.csv", index=False)


# In[8]:


create_java_only_dataset()


# In[9]:


def get_uuid(text):
    return text.split("/")[-1].split(".")[0]


# In[10]:


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


# In[11]:


def create_report_files():
    if not os.path.isdir(scratch_path + "partha9/Data/Report_Files/"):
        Path(scratch_path + "partha9/Data/Report_Files/").mkdir(parents=True, exist_ok=True)
        df = pd.read_csv(scratch_path + "partha9/Data/Java_Unified_Data_with_SHA.csv")
        for item in df.iterrows():
            uuid_name = item[1]['before_fix_uuid_file_path'].split("/")[-1].split(".")[0]
            file = open(scratch_path + "partha9/Data/Report_Files/{}.txt".format(uuid_name), "w")
            file.write(item[1]['title'] + " " + item[1]['description'])
            file.close()


# In[12]:


create_report_files()


# In[13]:


def convert_file_to_ast(file_path, parser):
    file = open(file_path, "r")
    file_content = file.read()
    tree = parser.parse(bytes(file_content, "utf-8"))
    return tree.root_node.sexp()


# In[14]:


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


# In[15]:


create_ast_files()


# In[16]:


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


# In[32]:

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
            before_fix_ast_path = scratch_path + "partha9/Data/AST_Files/" + get_uuid(
                rows['before_fix_uuid_file_path']) + ".txt"
            after_fix_ast_path = scratch_path + "partha9/Data/AST_Files/" + get_uuid(
                rows['after_fix_uuid_file_path']) + ".txt"
            report_files = scratch_path + "partha9/Data/Report_Files/" + get_uuid(
                rows['before_fix_uuid_file_path']) + ".txt"
        else:
            before_fix_ast_path = rows['before_fix_uuid_file_path'].map(
                lambda x: scratch_path + "partha9/Data/AST_Files/" + get_uuid(x) + ".txt").tolist()
            after_fix_ast_path = rows['after_fix_uuid_file_path'].map(
                lambda x: scratch_path + "partha9/Data/AST_Files/" + get_uuid(x) + ".txt").tolist()
            report_files = rows['before_fix_uuid_file_path'].map(
                lambda x: scratch_path + "partha9/Data/Report_Files/" + get_uuid(x) + ".txt").tolist()
        temp = file_reader(before_fix_ast_path, after_fix_ast_path, report_files)
        before, after = self.tokenizer.encode_plus(temp[1], truncation=True, max_length=1498)['input_ids'], \
                        self.tokenizer.encode_plus(temp[2], truncation=True, max_length=1498)['input_ids']
        start, end = find_difference(before, after)
        report_context = self.tokenizer.encode_plus(temp[0], temp[1], truncation=True, max_length=512, padding=True,
                                                    pad_to_multiple_of=512)
        return {'input_ids': torch.tensor(report_context['input_ids']),
                'attention_mask': torch.tensor(report_context['attention_mask']), 'start_positions': start,
                'end_positions': end}


# In[18]:


train_data, val_data = train_test_split(pd.read_csv(scratch_path + "partha9/Data/Java_Train_Data.csv"),
                                        test_size=0.125)

# In[19]:


before_fix_ast_paths = train_data['before_fix_uuid_file_path'].map(
    lambda x: scratch_path + "partha9/Data/AST_Files/" + get_uuid(x) + ".txt").tolist()
after_fix_ast_paths = train_data['after_fix_uuid_file_path'].map(
    lambda x: scratch_path + "partha9/Data/AST_Files/" + get_uuid(x) + ".txt").tolist()
report_files = train_data['before_fix_uuid_file_path'].map(
    lambda x: scratch_path + "partha9/Data/Report_Files/" + get_uuid(x) + ".txt").tolist()

# In[20]:


all_file_path = before_fix_ast_paths + report_files

# In[21]:


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

# In[22]:


# tokenizer = ByteLevelBPETokenizer(
#     "aster-vocab.json",
#     "aster-merges.txt",
# )
# tokenizer._tokenizer.post_processor = RobertaProcessing(
#     ("</s>", tokenizer.token_to_id("</s>")),
#     ("<s>", tokenizer.token_to_id("<s>")),
# )
# tokenizer.enable_truncation(max_length=3000)


# In[23]:


tokenizer = RobertaTokenizer(root_path + "/tokenizer/aster-vocab.json", root_path + "/tokenizer/aster-merges.txt")

# In[24]:


device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
# save point
save_at = 500
# load model
model = RobertaForQuestionAnswering.from_pretrained("/project/def-m2nagapp/partha9/Aster/PlainRobertaWithAst_Size_Extension" + "/train_output/" + "checkpoint-18000/")
train_dataset = BugDataset(dataframe=train_data, tokenizer=tokenizer)
# move model over to detected device
model.to(device)
# activate training mode of model
model.train()
# initialize adam optimizer with weight decay (reduces chance of overfitting)
optim = AdamW(model.parameters(), lr=5e-5)

# initialize data loader for training data
train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)

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
                scratch_path + "partha9/Aster/PlainRobertaWithAst_Size_Extension_QA/" + "/train_output/" + "CheckPoint-{}".format(
                    i))
