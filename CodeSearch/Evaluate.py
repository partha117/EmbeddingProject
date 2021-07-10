import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import jsonlines
from transformers import RobertaTokenizer, AutoConfig, AutoModel
import sys
import os
import numpy as np
from torch import nn as nn
sys.path.append(os.path.abspath("/home/partha9/EmbeddingProject"))
from Reformer import ElectraUtil
from Reformer.ElectraUtil import ElectraForPreTraining
import argparse
from scipy.spatial.distance import cdist

class BugDataset(Dataset):
        def __init__(self, file_path, tokenizer, limit):

            self.json_file = jsonlines.open(file_path, "r")
            self.file_iterator = iter(self.json_file)
            self.tokenizer = tokenizer
            self.limit = limit
            self.length = len(list(iter(jsonlines.open(file_path, "r"))))

        def __len__(self):
            return self.length

        def __getitem__(self, idx):
            data = next(self.file_iterator)
            code, description = data['code'], data['docstring']
            return np.array(self.tokenizer.encode_plus(code, truncation=True, max_length=self.limit, padding=True, pad_to_multiple_of=self.limit)['input_ids']), \
                   np.array(self.tokenizer.encode_plus(description, truncation=True, max_length=self.limit, padding=True,
                                              pad_to_multiple_of=self.limit)['input_ids'])



if __name__ == "__main__":
    root_path = ".."
    sys.modules['ElectraUtil'] = ElectraUtil
    # parser = argparse.ArgumentParser()
    # parser.add_argument('--electra', action='store_true', help="")
    # parser.add_argument('--state_dict', action='store_true', help="")
    # parser.add_argument('--config', default=None, type=str, help="")
    # args = parser.parse_args()
    # if args.electra:
    #     if args.state_dict:
    #         temp_config = AutoConfig.from_pretrained(args.config)
    #         temp_config.is_decoder = False
    #         temp_config.output_hidden_states = True
    #         full_base_model_dict = torch.load(args.model_path + args.checkpoint)
    #         full_base_model = ElectraForPreTraining(temp_config)
    #         full_base_model.load_state_dict(full_base_model_dict)
    #         full_base_model = full_base_model.electra
    #     else:
    #         full_base_model = torch.load(args.model_path + args.checkpoint).electra
    # else:
    #     full_base_model = AutoModel.from_pretrained(args.model_path + args.checkpoint)
    dev = "cpu"
    model = torch.load("../discriminator").electra
    model = model.eval().to(dev)
    batch_size = 10
    tokenizer = RobertaTokenizer(root_path + "/tokenizer/aster-vocab.json", root_path + "/tokenizer/aster-merges.txt")
    dataset = BugDataset(file_path="/home/partha/MY_DRIVES/ProgramFiles/EmbeddingProject/CodeSearch/java_test_0.jsonl", tokenizer=tokenizer, limit=1498)
    # dataloader = DataLoader(dataset, batch_size=int(args.batch_size), pin_memory=True, num_workers=2, sampler=sampler,
    #                         drop_last=True)
    dataloader = DataLoader(dataset, batch_size=batch_size, pin_memory=True, num_workers=1,
                            drop_last=True)
    st1 = "good"
    st2 = "bad"
    st3 = "better"
    st1, st2, st3 = torch.tensor(tokenizer.encode_plus(st1, truncation=True, max_length=1498, padding=True, pad_to_multiple_of=1498)['input_ids']), \
                    torch.tensor(tokenizer.encode_plus(st2, truncation=True, max_length=1498, padding=True,
                                                        pad_to_multiple_of=1498)['input_ids']), \
                    torch.tensor(tokenizer.encode_plus(st3, truncation=True, max_length=1498, padding=True,
                                                        pad_to_multiple_of=1498)['input_ids'])

    st1, st2, st3 = st1.to(dev), st2.to(dev), st3.to(dev)
    sts = torch.stack([st1,st2,st3])
    sts = model(sts)
    sts = sts[0].mean(1).detach().numpy()
    print(cdist(sts, sts, metric='cosine'))
    # code, description = next(iter(dataloader))
    # code = code.to(dev)
    # description = description.to(dev)
    # code_rep, description_rep = model(code), model(description)
    # code_rep = code_rep[0].mean(1).detach().numpy()
    # description_rep = description_rep[0].mean(1).detach().numpy()
    # similarity = 1 - cdist(code_rep, description_rep, metric='cosine')
    # diagonal_correct_elements = np.expand_dims(np.diag(similarity), axis=-1)
    # rank = np.sum(similarity > diagonal_correct_elements, axis=-1)
    # print(rank)
    # print(rank + 1)

