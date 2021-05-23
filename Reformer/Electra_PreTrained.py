import logging
import argparse
from collections import namedtuple
import torch
import os
from ElectraUtil import tie_weights, set_seed, Electra, LogitsAdapter, train, ElectraForPreTraining, return_sorted
from transformers import RobertaTokenizer, RobertaForMaskedLM, AdamW, AutoConfig, AutoModelForMaskedLM
from tokenizers import ByteLevelBPETokenizer
import pandas as pd
import re
from pathlib import Path
from sklearn.model_selection import train_test_split
from tree_sitter import Language, Parser
from joblib import Parallel, delayed
from torch.utils.data import Dataset, DataLoader


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
    if not os.path.isfile(scratch_path + "Data/Java_Unified_Data_with_SHA.csv"):
        df = pd.read_csv(scratch_path + "Data/Unified_Data_with_SHA.csv")
        df2 = df[df["language_name"] == 'Java']
        df2.reset_index(drop=True, inplace=True)
        df2.to_csv(scratch_path + "Data/Java_Unified_Data_with_SHA.csv", index=False)


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
    if not os.path.isdir(scratch_path + "Data/Report_Files/"):
        Path(scratch_path + "Data/Report_Files/").mkdir(parents=True, exist_ok=True)
        df = pd.read_csv(scratch_path + "Data/Java_Unified_Data_with_SHA.csv")
        for item in df.iterrows():
            uuid_name = item[1]['before_fix_uuid_file_path'].split("/")[-1].split(".")[0]
            file = open(scratch_path + "Data/Report_Files/{}.txt".format(uuid_name), "w")
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
    if not os.path.isdir(scratch_path + "Data/AST_Files/"):
        Path(scratch_path + "Data/AST_Files/").mkdir(parents=True, exist_ok=True)
        df = pd.read_csv(scratch_path + "Data/Java_Unified_Data_with_SHA.csv")
        path = scratch_path + ""
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
            self.dataset = pd.read_csv(file_path)# .sample(frac=0.315, random_state=13)
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
            before_fix_ast_path = scratch_path + "Data/AST_Files/" + get_uuid(
                rows['before_fix_uuid_file_path']) + ".txt"
            report_files = scratch_path + "Data/Report_Files/" + get_uuid(
                rows['before_fix_uuid_file_path']) + ".txt"
        else:
            before_fix_ast_path = rows['before_fix_uuid_file_path'].map(
                lambda x: scratch_path + "Data/AST_Files/" + get_uuid(x) + ".txt").tolist()
            report_files = rows['before_fix_uuid_file_path'].map(
                lambda x: scratch_path + "Data/Report_Files/" + get_uuid(x) + ".txt").tolist()
        temp = file_reader(before_fix_ast_path, report_files)
        return \
            self.tokenizer.encode_plus(temp, truncation=True, max_length=2048, padding=True, pad_to_multiple_of=2048)



if __name__ == "__main__":
    os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
    parser = argparse.ArgumentParser()

    ## Required parameters
    parser.add_argument("--data_dir", default=None, type=str,
                        help="The input data dir. Should contain the .tsv files (or other data files) for the task.")
    #     parser.add_argument("--model_type", default=None, type=str, required=True,
    #                         help="Model type selected in the list: " + ", ".join(MODEL_CLASSES.keys()))
    parser.add_argument("--model_name_or_path", default=None, type=str,
                        help="Path to pre-trained model or shortcut name")
    #     parser.add_argument("--task_name", default='codesearch', type=str, required=True,
    #                         help="The name of the task to train selected in the list: " + ", ".join(processors.keys()))
    parser.add_argument("--output_dir", default=None, type=str, required=True,
                        help="The output directory where the model predictions and checkpoints will be written.")

    ## Other parameters
    parser.add_argument("--config_name", default="", type=str,
                        help="Pretrained config name or path if not the same as model_name")
    parser.add_argument("--tokenizer_name", default=".", type=str,
                        help="Pretrained tokenizer name or path if not the same as model_name")
    parser.add_argument("--cache_dir", default="", type=str,
                        help="Where do you want to store the pre-trained models downloaded from s3")
    parser.add_argument("--max_seq_length", default=512, type=int,
                        help="The maximum total input sequence length after tokenization. Sequences longer "
                             "than this will be truncated, sequences shorter will be padded.")
    parser.add_argument("--do_train", action='store_true',
                        help="Whether to run training.")
    parser.add_argument("--do_eval", action='store_true',
                        help="Whether to run eval on the dev set.")
    parser.add_argument("--do_predict", action='store_true',
                        help="Whether to run predict on the test set.")
    parser.add_argument("--evaluate_during_training", action='store_true',
                        help="Rul evaluation during training at each logging step.")
    parser.add_argument("--do_lower_case", action='store_true',
                        help="Set this flag if you are using an uncased model.")

    parser.add_argument("--per_gpu_train_batch_size", default=8, type=int,
                        help="Batch size per GPU/CPU for training.")
    parser.add_argument("--per_gpu_eval_batch_size", default=8, type=int,
                        help="Batch size per GPU/CPU for evaluation.")
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument("--learning_rate", default=5e-5, type=float,
                        help="The initial learning rate for Adam.")
    parser.add_argument("--weight_decay", default=0.0, type=float,
                        help="Weight deay if we apply some.")
    parser.add_argument("--adam_epsilon", default=1e-8, type=float,
                        help="Epsilon for Adam optimizer.")
    parser.add_argument("--max_grad_norm", default=1.0, type=float,
                        help="Max gradient norm.")
    parser.add_argument("--num_train_epochs", default=3.0, type=float,
                        help="Total number of training epochs to perform.")
    parser.add_argument("--max_steps", default=-1, type=int,
                        help="If > 0: set total number of training steps to perform. Override num_train_epochs.")
    parser.add_argument("--warmup_steps", default=0, type=int,
                        help="Linear warmup over warmup_steps.")

    parser.add_argument('--logging_steps', type=int, default=50,
                        help="Log every X updates steps.")
    parser.add_argument('--save_steps', type=int, default=50,
                        help="Save checkpoint every X updates steps.")
    parser.add_argument("--eval_all_checkpoints", action='store_true',
                        help="Evaluate all checkpoints starting with the same prefix as model_name ending and ending with step number")
    parser.add_argument("--no_cuda", action='store_true',
                        help="Avoid using CUDA when available")
    parser.add_argument('--overwrite_output_dir', action='store_true',
                        help="Overwrite the content of the output directory")
    parser.add_argument('--overwrite_cache', action='store_true',
                        help="Overwrite the cached training and evaluation sets")
    parser.add_argument('--seed', type=int, default=13,
                        help="random seed for initialization")

    parser.add_argument('--fp16', action='store_true',
                        help="Whether to use 16-bit (mixed) precision (through NVIDIA apex) instead of 32-bit")
    parser.add_argument('--fp16_opt_level', type=str, default='O1',
                        help="For fp16: Apex AMP optimization level selected in ['O0', 'O1', 'O2', and 'O3']."
                             "See details at https://nvidia.github.io/apex/amp.html")
    parser.add_argument("--local_rank", type=int, default=-1,
                        help="For distributed training: local_rank")
    parser.add_argument('--server_ip', type=str, default='', help="For distant debugging.")
    parser.add_argument('--server_port', type=str, default='', help="For distant debugging.")
    parser.add_argument("--train_file", default="train_top10_concat.tsv", type=str,
                        help="train file")
    parser.add_argument("--dev_file", default="shared_task_dev_top10_concat.tsv", type=str,
                        help="dev file")
    parser.add_argument("--test_file", default="shared_task_dev_top10_concat.tsv", type=str,
                        help="test file")
    parser.add_argument("--pred_model_dir", default=None, type=str,
                        help='model for prediction')
    parser.add_argument("--test_result_dir", default='test_results.tsv', type=str,
                        help='path to store test result')

    parser.add_argument("--gen_model_name_or_path", default='distilroberta-base', type=str,
                        help='path to get generator model')
    parser.add_argument("--dis_model_name_or_path", default='.', type=str,
                        help='path to get discriminator model')
    args = parser.parse_args()


    logger = logging.getLogger(__name__)
    Results = namedtuple('Results', [
        'loss',
        'mlm_loss',
        'disc_loss',
        'gen_acc',
        'disc_acc',
        'disc_labels',
        'disc_predictions'
    ])

    is_cedar = False
    build_lib()
    scratch_path = "/scratch/"
    if not is_cedar:
        scratch_path += "partha9/"
    root_path = "/project/def-m2nagapp/partha9/Aster/Reformer_Electra/"
    Path(root_path).mkdir(parents=True, exist_ok=True)
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    create_java_only_dataset()
    create_report_files()
    create_ast_files()
    train_data, val_data = train_test_split(pd.read_csv(scratch_path + "Data/Java_Train_Data.csv"),
                                            test_size=0.125)
    before_fix_ast_paths = train_data['before_fix_uuid_file_path'].map(
        lambda x: scratch_path + "Data/AST_Files/" + get_uuid(x) + ".txt").tolist()
    after_fix_ast_paths = train_data['after_fix_uuid_file_path'].map(
        lambda x: scratch_path + "Data/AST_Files/" + get_uuid(x) + ".txt").tolist()
    report_files = train_data['before_fix_uuid_file_path'].map(
        lambda x: scratch_path + "Data/Report_Files/" + get_uuid(x) + ".txt").tolist()
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
    temp_dataset = BugDataset(file_path=scratch_path + "Data/Java_Train_Data.csv",tokenizer=tokenizer)
    latest_checkpoint, start_from = return_sorted(args.output_dir,lambda x: int(x.split("checkpoint-")[-1]) if re.search(r'checkpoint-\d+', x) else None, return_last=True)
    latest_checkpoint = os.path.join(args.output_dir, latest_checkpoint) if latest_checkpoint is not None else latest_checkpoint


    # Setup distant debugging if needed
    if args.server_ip and args.server_port:
        # Distant debugging - see https://code.visualstudio.com/docs/python/debugging#_attach-to-a-local-script
        import ptvsd

        print("Waiting for debugger attach")
        ptvsd.enable_attach(address=(args.server_ip, args.server_port), redirect_output=True)
        ptvsd.wait_for_attach()

    # Setup CUDA, GPU & distributed training
    if args.local_rank == -1 or args.no_cuda:
        device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
        args.n_gpu = torch.cuda.device_count()
    else:  # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        torch.distributed.init_process_group(backend='nccl')
        args.n_gpu = 1
    args.device = device

    # Setup logging
    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                        datefmt='%m/%d/%Y %H:%M:%S',
                        level=logging.INFO if args.local_rank in [-1, 0] else logging.WARN)
    logger.warning("Process rank: %s, device: %s, n_gpu: %s, distributed training: %s, 16-bits training: %s",
                   args.local_rank, device, args.n_gpu, bool(args.local_rank != -1), args.fp16)

    # Set seed
    set_seed(args)

    # Load pretrained model and tokenizer
    if args.local_rank not in [-1, 0]:
        torch.distributed.barrier()  # Make sure only the first process in distributed training will download model & vocab

    args.start_epoch = 0
    args.start_step = 0
    checkpoint_last = os.path.join(args.output_dir, latest_checkpoint) if latest_checkpoint is not None else os.path.join(args.output_dir, 'checkpoint-last')
    if os.path.exists(checkpoint_last) and os.listdir(checkpoint_last):
        args.gen_model_name_or_path = os.path.join(checkpoint_last, 'generator')
        logger.info("Generator Last Checkpoint {}".format(args.gen_model_name_or_path))
        args.dis_model_name_or_path = os.path.join(checkpoint_last, 'discriminator')

        idx_file = os.path.join(checkpoint_last, 'idx_file.txt')
        with open(idx_file, encoding='utf-8') as idxf:
            args.start_epoch = int(idxf.readlines()[0].strip()) + 1

        step_file = os.path.join(checkpoint_last, 'step_file.txt')
        if os.path.exists(step_file):
            with open(step_file, encoding='utf-8') as stepf:
                args.start_step = int(stepf.readlines()[0].strip())

        logger.info("reload model from {}, resume from {} epoch".format(checkpoint_last, args.start_epoch))
    if args.tokenizer_name:
        tokenizer_name = args.tokenizer_name
    elif args.model_name_or_path:
        tokenizer_name = 'roberta-base'
    tokenizer = RobertaTokenizer.from_pretrained(tokenizer_name)
    logger.info("Tokenizer loaded {}".format(tokenizer_name))
    temp_config = AutoConfig.from_pretrained(args.dis_model_name_or_path)
    temp_config.is_decoder = False
    generator = AutoModelForMaskedLM.from_pretrained(args.gen_model_name_or_path) if latest_checkpoint is not None else AutoModelForMaskedLM.from_config(config=temp_config)
    for param in generator.reformer.parameters():
        param.requires_grad = False
    logger.info("Generator {}".format(generator.config._name_or_path))
    discriminator = ElectraForPreTraining(temp_config)
    logger.info(
        "Generator {} and Discriminator {}".format(generator.config._name_or_path, discriminator.config._name_or_path))

    tie_weights(generator, discriminator)

    if args.local_rank == 0:
        torch.distributed.barrier()  # Make sure only the first process in distributed training will download model & vocab

    # Distributed and parallel training
    pad_token_id = tokenizer.pad_token_id
    mask_token_id = tokenizer.mask_token_id
    cls_token_id = tokenizer.cls_token_id
    sep_token_id = tokenizer.sep_token_id
    vocab_size = tokenizer.vocab_size
    print("Vocab size", vocab_size)

    model = Electra(
        LogitsAdapter(generator),
        discriminator,
        num_tokens=vocab_size,
        mask_token_id=mask_token_id,
        pad_token_id=pad_token_id,
        mask_prob=0.15,
        mask_ignore_token_ids=[cls_token_id, sep_token_id],
        random_token_prob=0.1).to(device)
    logger.info("Model Built!")

    # Prepare optimizer and schedule (linear warmup and decay)

    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
         'weight_decay': args.weight_decay},
        {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)

    optimizer_last = os.path.join(checkpoint_last, 'optimizer.pt')
    if os.path.exists(optimizer_last):
        optimizer.load_state_dict(torch.load(optimizer_last))

    if args.fp16:
        try:
            from apex import amp
        except ImportError:
            raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use fp16 training.")
        model, optimizer = amp.initialize(model, optimizer, opt_level=args.fp16_opt_level)

    if args.n_gpu > 1:
        model = torch.nn.DataParallel(model)

    if args.local_rank != -1:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.local_rank],
                                                          output_device=args.local_rank,
                                                          find_unused_parameters=True)

    logger.info("Training/evaluation parameters %s", args)

    # Training
    if args.do_train:
        train_dataset = temp_dataset # get_dataset(args.train_file, args.max_seq_length, tokenizer)
        eval_dataset = temp_dataset # get_dataset(args.dev_file, args.max_seq_length, tokenizer)
        if not os.path.exists(args.output_dir) and args.local_rank in [-1, 0]:
            os.makedirs(args.output_dir)
        global_step = train(args, train_dataset, eval_dataset, model, generator, discriminator, tokenizer, optimizer,
                            pad_token_id, logger)
        # logger.info(" global_step = %s, average loss = %s", global_step, tr_loss)

    # Saving best-practices: if you use defaults names for the model, you can reload it using from_pretrained()
    if args.do_train and (args.local_rank == -1 or torch.distributed.get_rank() == 0):
        # Create output directory if needed
        if not os.path.exists(args.output_dir) and args.local_rank in [-1, 0]:
            os.makedirs(args.output_dir)
        generator_path = os.path.join(args.output_dir, 'generator')
        discriminator_path = os.path.join(args.output_dir, 'discriminator')

        logger.info("Saving model checkpoint to %s", args.output_dir)
        # Save a trained model, configuration and tokenizer using `save_pretrained()`.
        # They can then be reloaded using `from_pretrained()`
        gen_model_to_save = generator.module if hasattr(generator,
                                                        'module') else generator  # Take care of distributed/parallel training
        disc_model_to_save = discriminator.module if hasattr(discriminator,
                                                             'module') else discriminator
        gen_model_to_save.reformer.save_pretrained(generator_path)
        disc_model_to_save.electra.save_pretrained(discriminator_path)
        torch.save(args, os.path.join(args.output_dir, 'training_args.bin'))
