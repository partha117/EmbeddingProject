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
from transformers import DataCollatorForLanguageModeling, RobertaConfig, ReformerConfig, XLNetConfig, XLMConfig, XLMRobertaConfig
from transformers import Trainer, TrainingArguments
from transformers import AdamW, RobertaModel, AutoModel,RobertaTokenizer, AutoModelForMaskedLM, RobertaForMaskedLM
from joblib import Parallel, delayed


# In[4]:

scratch_path = "/scratch/"
root_path = "/project/def-m2nagapp/partha9/Aster/PlainRobertaWithAst_Size_Extension/"
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
        df2 = df[df["language_name"]=='Java']
        df2.reset_index(drop=True,inplace=True)
        df2.to_csv("Data/Java_Unified_Data_with_SHA.csv",index=False)


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
            return " " # note: a space and not an empty string
        else:
            return s
    pattern = re.compile(
        r'//.*?$|/\*.*?\*/|\'(?:\\.|[^\\\'])*\'|"(?:\\.|[^\\"])*"',
        re.DOTALL | re.MULTILINE
    )
    temp=[]
    for x in re.sub(pattern, replacer, source).split('\n'):
        if x.strip()!="":
            temp.append(x)
    return '\n'.join(temp)


# In[11]:


def create_report_files():
    if not os.path.isdir(scratch_path + "partha9/Data/Report_Files/"):
        Path(scratch_path + "partha9/Data/Report_Files/").mkdir(parents=True, exist_ok=True)
        df = pd.read_csv(scratch_path + "partha9/Data/Java_Unified_Data_with_SHA.csv")
        for item in df.iterrows():
            uuid_name = item[1]['before_fix_uuid_file_path'].split("/")[-1].split(".")[0]
            file = open (scratch_path + "partha9/Data/Report_Files/{}.txt".format(uuid_name),"w")
            file.write(item[1]['title'] + " " + item[1]['description'])
            file.close()


# In[12]:


create_report_files()


# In[13]:


def convert_file_to_ast(file_path, parser):
    file = open(file_path,"r")
    file_content = file.read()
    tree = parser.parse(bytes(file_content,"utf-8"))
    return tree.root_node.sexp()


# In[14]:



def save_file(path, item):
    JAVA_LANGUAGE = Language('/project/def-m2nagapp/partha9/Data/build/my-languages.so', 'java')
    parser = Parser()
    parser.set_language(JAVA_LANGUAGE)
    before_fix_uuid_name = item[1]['before_fix_uuid_file_path'].split("/")[-1].split(".")[0]
    before_fix_file = open (path + "Data/AST_Files/{}.txt".format(before_fix_uuid_name),"w")
    before_fix_file.write(convert_file_to_ast(path + item[1]['before_fix_uuid_file_path'],parser))
    before_fix_file.close()

    after_fix_uuid_name = item[1]['after_fix_uuid_file_path'].split("/")[-1].split(".")[0]
    after_fix_file = open (path + "Data/AST_Files/{}.txt".format(after_fix_uuid_name),"w")
    after_fix_file.write(convert_file_to_ast(path + item[1]['after_fix_uuid_file_path'],parser))
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


def file_reader(ast_paths,report_paths):
    if not isinstance(ast_paths, str):
        accumulate  = []
        for ast, report in zip(ast_paths,report_paths):
            temp = ""
            with open(report,"r") as file:
                temp += file.read()
            with open(ast,"r") as file:
                temp += file.read()  
            accumulate.append(temp)
    else:
        acuumulate = None
        temp = ""
        with open(report_paths,"r") as file:
            temp += file.read()
        with open(ast_paths,"r") as file:
            temp += file.read() 
        accumulate = temp
    return accumulate


# In[32]:


class BugDataset(Dataset):

    def __init__(self, file_path=None,dataframe=None,tokenizer=None):
        if file_path is not None:
            self.dataset = pd.read_csv(file_path)
        else:
            self.dataset =dataframe
        self.tokenizer = tokenizer
    def __len__(self):
        return len(self.dataset)


    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        rows = self.dataset.iloc[idx,:]
        if isinstance(idx, int):
            before_fix_ast_path = scratch_path + "partha9/Data/AST_Files/" + get_uuid(rows['before_fix_uuid_file_path']) + ".txt"
            report_files = scratch_path + "partha9/Data/Report_Files/" + get_uuid(rows['before_fix_uuid_file_path']) + ".txt"
        else:
            before_fix_ast_path = rows['before_fix_uuid_file_path'].map(lambda x:scratch_path + "partha9/Data/AST_Files/" + get_uuid(x) + ".txt").tolist()
            report_files = rows['before_fix_uuid_file_path'].map(lambda x:scratch_path + "partha9/Data/Report_Files/" + get_uuid(x) + ".txt").tolist()
        temp = file_reader(before_fix_ast_path, report_files)
        return  self.tokenizer.encode_plus(temp,truncation=True, max_length=1498)['input_ids']


# In[18]:


train_data,val_data = train_test_split(pd.read_csv(scratch_path + "partha9/Data/Java_Train_Data.csv"),test_size=0.125)


# In[19]:


before_fix_ast_paths = train_data['before_fix_uuid_file_path'].map(lambda x:scratch_path + "partha9/Data/AST_Files/" + get_uuid(x) + ".txt").tolist()
after_fix_ast_paths = train_data['after_fix_uuid_file_path'].map(lambda x:scratch_path + "partha9/Data/AST_Files/" + get_uuid(x) + ".txt").tolist()
report_files = train_data['before_fix_uuid_file_path'].map(lambda x:scratch_path + "partha9/Data/Report_Files/" + get_uuid(x) + ".txt").tolist()


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


tokenizer = RobertaTokenizer(root_path + "/tokenizer/aster-vocab.json",root_path + "/tokenizer/aster-merges.txt")


# In[24]:


temp_dataset = BugDataset(scratch_path + "partha9/Data/Java_Train_Data.csv")


# In[25]:


temp_dataloader = DataLoader(temp_dataset, batch_size=4, num_workers=1)


# In[26]:


data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer, mlm=True, mlm_probability=0.15
)


# In[38]:


Path(root_path + "/train_output/").mkdir(parents=True, exist_ok=True)
training_args = TrainingArguments(
    output_dir=root_path + "/train_output/",
    overwrite_output_dir=True,
    num_train_epochs=4,
    per_device_train_batch_size=2,
    save_steps=500,
    save_total_limit=4,
    dataloader_drop_last = True

)


# In[28]:


# RobertaConfig, ReformerConfig, XLNetConfig, XLMConfig, XLMRobertaConfig


# In[29]:


config = RobertaConfig.from_pretrained("/project/def-m2nagapp/partha9/model_cache/roberta_1500_config/") # RobertaConfig.from_pretrained("roberta-base",max_position_embeddings=3500)


# In[39]:


model = RobertaForMaskedLM(config=config) # RobertaForMaskedLM.from_pretrained(root_path + "/train_output/" + "checkpoint-6500/") #RobertaForMaskedLM(config=config)
#model.spread_on_devices()
trainer = Trainer(
    model= model,
    args= training_args,
    data_collator= data_collator,
    train_dataset= BugDataset(dataframe=train_data,tokenizer=tokenizer),
#     eval_dataset= BugDataset(dataframe=val_data,tokenizer=tokenizer)
)


# In[44]:


trainer


# In[40]:


trainer.train()

