from transformers import DataCollatorForLanguageModeling, RobertaConfig, ReformerConfig, XLNetConfig, XLMConfig, \
    XLMRobertaConfig, AutoConfig, AutoTokenizer
from transformers import AdamW, RobertaModel, AutoModel, RobertaTokenizer, AutoModelForMaskedLM, RobertaForMaskedLM, \
    XLNetLMHeadModel, ReformerForMaskedLM, ReformerForQuestionAnswering
import argparse


def save_config(name, save_name, max_embedding=None, axial_pos_shape=None):
    params = {}
    if max_embedding:
        params = {"max_position_embeddings": int(max_embedding)}
    if axial_pos_shape:
        params = {"axial_pos_shape": (int(axial_pos_shape[0]), int(axial_pos_shape[1]))}
    print(params)
    if len(params.keys()) > 0:
        config = AutoConfig.from_pretrained(name, **params)
    else:
        config = AutoConfig.from_pretrained(name)
    config.save_pretrained("/project/6033386/partha9/model_cache/{}_config".format(save_name))


def save_tokenizer(name, save_name):
    tokenizer = AutoTokenizer.from_pretrained(name)
    tokenizer.save_pretrained("/project/6033386/partha9/model_cache/{}_tokenizer".format(save_name))


def save_model(name, save_name):
    model = AutoModel.from_pretrained(name)
    model.save_pretrained("/project/6033386/partha9/model_cache/{}_model".format(save_name))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--task_name", type=str,
                        help="Type of download")
    parser.add_argument("--name", type=str,
                        help="Name to download")
    parser.add_argument("--save_name", type=str,
                        help="Name to save")
    parser.add_argument("--max_embedding", default=None,
                        help="Maximum embedding size")
    parser.add_argument("--axial_pos_shape", default=None,nargs='+',
                        help="Axial position shape")
    args = parser.parse_args()
    print(args)
    if args.task_name == 'config':
        save_config(args.name, args.save_name, args.max_embedding, args.axial_pos_shape)
    elif args.task_name == 'tokenizer':
        save_tokenizer(args.name, args.save_name)
    elif args.task_name == 'model':
        save_model(args.name, args.save_name)
