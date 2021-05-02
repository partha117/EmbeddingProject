import argparse
import glob
import logging
import os
import random
import math
from functools import reduce
from collections import namedtuple

import torch
from torch import nn
import torch.nn.functional as F

from tqdm import tqdm, trange
from electra_pytorch import Electra
import torch.nn as nn
import numpy as np
import torch
import pandas as pd
from transformers.activations import get_activation

from torch.utils.data import (DataLoader, RandomSampler, SequentialSampler,
                              TensorDataset)
from torch.utils.data.distributed import DistributedSampler
from tensorboardX import SummaryWriter
from tqdm import tqdm, trange

from transformers import RobertaTokenizer, RobertaForMaskedLM, RobertaModel, BertPreTrainedModel, AutoConfig, \
    get_linear_schedule_with_warmup, AdamW

# from utils import (compute_metrics, convert_examples_to_features,
#                         output_modes, processors)

logger = logging.getLogger(__name__)

# MODEL_CLASSES = {'roberta': (RobertaConfig, RobertaForSequenceClassification, RobertaTokenizer)}
Results = namedtuple('Results', [
    'loss',
    'mlm_loss',
    'disc_loss',
    'gen_acc',
    'disc_acc',
    'disc_labels',
    'disc_predictions'
])


# helpers

def log(t, eps=1e-9):
    return torch.log(t + eps)


def gumbel_noise(t):
    noise = torch.zeros_like(t).uniform_(0, 1)
    return -log(-log(noise))


def gumbel_sample(t, temperature=1.):
    return ((t / temperature) + gumbel_noise(t)).argmax(dim=-1)


def prob_mask_like(t, prob):
    return torch.zeros_like(t).float().uniform_(0, 1) < prob


def mask_with_tokens(t, token_ids):
    init_no_mask = torch.full_like(t, False, dtype=torch.bool)
    mask = reduce(lambda acc, el: acc | (t == el), token_ids, init_no_mask)
    return mask


def get_mask_subset_with_prob(mask, prob):
    batch, seq_len, device = *mask.shape, mask.device
    max_masked = math.ceil(prob * seq_len)

    num_tokens = mask.sum(dim=-1, keepdim=True)
    mask_excess = (mask.cumsum(dim=-1) > (num_tokens * prob).ceil())
    mask_excess = mask_excess[:, :max_masked]

    rand = torch.rand((batch, seq_len), device=device).masked_fill(~mask, -1e9)
    _, sampled_indices = rand.topk(max_masked, dim=-1)
    sampled_indices = (sampled_indices + 1).masked_fill_(mask_excess, 0)

    new_mask = torch.zeros((batch, seq_len + 1), device=device)
    new_mask.scatter_(-1, sampled_indices, 1)
    return new_mask[:, 1:].bool()


# hidden layer extractor class, for magically adding adapter to language model to be pretrained

class HiddenLayerExtractor(nn.Module):
    def __init__(self, net, layer=-2):
        super().__init__()
        self.net = net
        self.layer = layer

        self.hidden = None
        self.hook_registered = False

    def _find_layer(self):
        if type(self.layer) == str:
            modules = dict([*self.net.named_modules()])
            return modules.get(self.layer, None)
        elif type(self.layer) == int:
            children = [*self.net.children()]
            return children[self.layer]
        return None

    def _hook(self, _, __, output):
        self.hidden = output

    def _register_hook(self):
        layer = self._find_layer()
        assert layer is not None, f'hidden layer ({self.layer}) not found'
        handle = layer.register_forward_hook(self._hook)
        self.hook_registered = True

    def forward(self, x):
        if self.layer == -1:
            return self.net(x)

        if not self.hook_registered:
            self._register_hook()

        _ = self.net(x)
        hidden = self.hidden
        self.hidden = None
        assert hidden is not None, f'hidden layer {self.layer} never emitted an output'
        return hidden


# main electra class

class Electra(nn.Module):
    def __init__(
            self,
            generator,
            discriminator,
            *,
            num_tokens=None,
            discr_dim=-1,
            discr_layer=-1,
            mask_prob=0.15,
            replace_prob=0.85,
            random_token_prob=0.,
            mask_token_id=2,
            pad_token_id=0,
            mask_ignore_token_ids=[],
            disc_weight=50.,
            gen_weight=1.,
            temperature=1.):
        super().__init__()

        self.generator = generator
        self.discriminator = discriminator

        if discr_dim > 0:
            self.discriminator = nn.Sequential(
                HiddenLayerExtractor(discriminator, layer=discr_layer),
                nn.Linear(discr_dim, 1)
            )

        # mlm related probabilities
        self.mask_prob = mask_prob
        self.replace_prob = replace_prob

        self.num_tokens = num_tokens
        self.random_token_prob = random_token_prob

        # token ids
        self.pad_token_id = pad_token_id
        self.mask_token_id = mask_token_id
        self.mask_ignore_token_ids = set([*mask_ignore_token_ids, pad_token_id])

        # sampling temperature
        self.temperature = temperature

        # loss weights
        self.disc_weight = disc_weight
        self.gen_weight = gen_weight

    def forward(self, input, **kwargs):
        b, t = input.shape

        replace_prob = prob_mask_like(input, self.replace_prob)

        # do not mask [pad] tokens, or any other tokens in the tokens designated to be excluded ([cls], [sep])
        # also do not include these special tokens in the tokens chosen at random
        no_mask = mask_with_tokens(input, self.mask_ignore_token_ids)
        mask = get_mask_subset_with_prob(~no_mask, self.mask_prob)

        # get mask indices
        mask_indices = torch.nonzero(mask, as_tuple=True)

        # mask input with mask tokens with probability of `replace_prob` (keep tokens the same with probability 1 - replace_prob)
        masked_input = input.clone().detach()

        # if random token probability > 0 for mlm
        if self.random_token_prob > 0:
            assert self.num_tokens is not None, 'Number of tokens (num_tokens) must be passed to Electra for randomizing tokens during masked language modeling'

            random_token_prob = prob_mask_like(input, self.random_token_prob)
            random_tokens = torch.randint(0, self.num_tokens, input.shape, device=input.device)
            random_no_mask = mask_with_tokens(random_tokens, self.mask_ignore_token_ids)
            random_token_prob &= ~random_no_mask
            random_indices = torch.nonzero(random_token_prob, as_tuple=True)
            masked_input[random_indices] = random_tokens[random_indices]

        # [mask] input
        masked_input = masked_input.masked_fill(mask * replace_prob, self.mask_token_id)

        # set inverse of mask to padding tokens for labels
        gen_labels = input.masked_fill(~mask, self.pad_token_id)
        # get generator output and get mlm loss
        logits = self.generator(masked_input, **kwargs)
        #         print("PRINTT",gen_labels,logits,gen_labels.shape,logits.shape)

        mlm_loss = F.cross_entropy(
            logits.transpose(1, 2),
            gen_labels,
            ignore_index=self.pad_token_id
        )

        # use mask from before to select logits that need sampling
        sample_logits = logits[mask_indices]

        # sample
        sampled = gumbel_sample(sample_logits, temperature=self.temperature)

        # scatter the sampled values back to the input
        disc_input = input.clone()
        disc_input[mask_indices] = sampled.detach()

        # generate discriminator labels, with replaced as True and original as False
        disc_labels = (input != disc_input).float().detach()
        #         print("disc_labels shape", disc_labels.shape)
        # get discriminator predictions of replaced / original
        non_padded_indices = torch.nonzero(input != self.pad_token_id, as_tuple=True)

        # get discriminator output and binary cross entropy loss
        disc_logits = self.discriminator(disc_input, **kwargs)
        #         print("disc_logits shape",disc_logits.shape)
        disc_logits = disc_logits.reshape_as(disc_labels)

        disc_loss = F.binary_cross_entropy_with_logits(
            disc_logits[non_padded_indices],
            disc_labels[non_padded_indices]
        )

        # gather metrics
        with torch.no_grad():
            gen_predictions = torch.argmax(logits, dim=-1)
            disc_predictions = torch.round((torch.sign(disc_logits) + 1.0) * 0.5)
            gen_acc = (gen_labels[mask] == gen_predictions[mask]).float().mean()
            disc_acc = 0.5 * (disc_labels[mask] == disc_predictions[mask]).float().mean() + 0.5 * (
                        disc_labels[~mask] == disc_predictions[~mask]).float().mean()

        # return weighted sum of losses
        return self.gen_weight * mlm_loss + self.disc_weight * disc_loss, mlm_loss, disc_loss, gen_acc, disc_acc, disc_labels, disc_predictions


class ElectraDiscriminatorPredictions(nn.Module):

    def __init__(self, config):
        super().__init__()

        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.dense_prediction = nn.Linear(config.hidden_size, 1)
        self.config = config

    def forward(self, discriminator_hidden_states):
        hidden_states = self.dense(discriminator_hidden_states)
        hidden_states = get_activation(self.config.hidden_act)(hidden_states)
        logits = self.dense_prediction(hidden_states).squeeze(-1)
        return logits


# This is electra discriminator
# avaialable in huggingface
class ElectraForPreTraining(BertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.electra = RobertaModel(config)
        self.discriminator_predictions = ElectraDiscriminatorPredictions(config)
        self.init_weights()

    def forward(
            self,
            input_ids=None,
            attention_mask=None,
            token_type_ids=None,
            position_ids=None,
            head_mask=None,
            inputs_embeds=None,
            labels=None,
            output_attentions=None,
            output_hidden_states=None,
            return_dict=None,
    ):
        discriminator_hidden_states = self.electra(
            input_ids,
            attention_mask,
            token_type_ids,
            position_ids,
            head_mask,
            inputs_embeds,
            output_attentions,
            output_hidden_states,
            return_dict,
        )
        discriminator_sequence_output = discriminator_hidden_states[0]
        logits = self.discriminator_predictions(discriminator_sequence_output)
        return logits


def collate_batch(examples, pad_token_id):
    input_ids = torch.nn.utils.rnn.pad_sequence([example[0] for example in examples], batch_first=True,
                                                padding_value=pad_token_id)
    input_mask = torch.nn.utils.rnn.pad_sequence([example[1] for example in examples], batch_first=True,
                                                 padding_value=pad_token_id)
    return input_ids, input_mask


def tie_weights(generator, discriminator):
    generator.roberta.embeddings.word_embeddings = discriminator.electra.embeddings.word_embeddings
    generator.roberta.embeddings.position_embeddings = discriminator.electra.embeddings.position_embeddings
    generator.roberta.embeddings.token_type_embeddings = discriminator.electra.embeddings.token_type_embeddings


class LogitsAdapter(torch.nn.Module):
    def __init__(self, adaptee):
        super().__init__()
        self.adaptee = adaptee

    def forward(self, *args, **kwargs):
        return self.adaptee(*args, **kwargs)[0]


def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)


def train(args, train_dataset, eval_dataset, model, generator, discriminator, tokenizer, optimizer, pad_token_id):
    """ Train the model """
    if args.local_rank in [-1, 0]:
        tb_writer = SummaryWriter()

    args.train_batch_size = args.per_gpu_train_batch_size * max(1, args.n_gpu)
    train_sampler = RandomSampler(train_dataset) if args.local_rank == -1 else DistributedSampler(train_dataset)
    train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=args.train_batch_size,
                                  collate_fn=lambda b: collate_batch(b, pad_token_id))
    if args.max_steps > 0:
        t_total = args.max_steps
        args.num_train_epochs = args.max_steps // (len(train_dataloader) // args.gradient_accumulation_steps) + 1
    else:
        t_total = len(train_dataloader) // args.gradient_accumulation_steps * args.num_train_epochs

    scheduler = get_linear_schedule_with_warmup(optimizer, args.warmup_steps, t_total)

    checkpoint_last = os.path.join(args.output_dir, 'checkpoint-last')
    scheduler_last = os.path.join(checkpoint_last, 'scheduler.pt')
    if os.path.exists(scheduler_last):
        scheduler.load_state_dict(torch.load(scheduler_last))

    # Train!
    logger.info("***** Running training *****")
    logger.info("  Num examples = %d", len(train_dataset))
    logger.info("  Num Epochs = %d", args.num_train_epochs)
    logger.info("  Instantaneous batch size per GPU = %d", args.per_gpu_train_batch_size)
    logger.info("  Total train batch size (w. parallel, distributed & accumulation) = %d",
                args.train_batch_size * args.gradient_accumulation_steps * (
                    torch.distributed.get_world_size() if args.local_rank != -1 else 1))
    logger.info("  Gradient Accumulation steps = %d", args.gradient_accumulation_steps)
    logger.info("  Total optimization steps = %d", t_total)

    global_step = args.start_step
    tr_loss = 0.0
    tr_loss_mlm = 0.0
    tr_loss_disc = 0.0
    tr_acc_gen = 0.0
    tr_acc_disc = 0.0

    best_acc = 0.0
    model.zero_grad()
    train_iterator = trange(args.start_epoch, int(args.num_train_epochs), desc="Epoch",
                            disable=args.local_rank not in [-1, 0])
    set_seed(args)  # Added here for reproductibility (even between python 2 and 3)
    model.train()
    for idx, _ in enumerate(train_iterator):
        tr_loss, tr_loss_mlm, tr_loss_disc, tr_acc_gen, tr_acc_disc = 0.0, 0.0, 0.0, 0.0, 0.0
        nb_train_steps = 0
        for step, batch in enumerate(train_dataloader):

            batch = tuple(t.to(args.device) for t in batch)
            loss, loss_mlm, loss_disc, acc_gen, acc_disc, disc_labels, disc_pred = model(batch[0],
                                                                                         attention_mask=batch[1])

            if args.n_gpu > 1:
                loss = loss.mean()  # mean() to average on multi-gpu parallel training
                loss_mlm = loss_mlm.mean()
                loss_disc = loss_disc.mean()
                acc_gen = acc_gen.mean()
                acc_disc = acc_disc.mean()
            if args.gradient_accumulation_steps > 1:
                loss = loss / args.gradient_accumulation_steps

            if args.fp16:
                try:
                    from apex import amp
                except ImportError:
                    raise ImportError(
                        "Please install apex from https://www.github.com/nvidia/apex to use fp16 training.")
                with amp.scale_loss(loss, optimizer) as scaled_loss:
                    scaled_loss.backward()
                torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer), args.max_grad_norm)
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)

            tr_loss += loss.item()
            tr_loss_mlm += loss_mlm.item()
            tr_loss_disc += loss_disc.item()
            tr_acc_gen += acc_gen.item()
            tr_acc_disc += acc_disc.item()
            nb_train_steps += 1
            if (step + 1) % args.gradient_accumulation_steps == 0:
                optimizer.step()
                scheduler.step()  # Update learning rate schedule
                model.zero_grad()
                global_step += 1

                if args.local_rank in [-1, 0] and args.logging_steps > 0 and global_step % args.logging_steps == 0:
                    logger.info(" Training: Epoch=%s, Step = %s", idx, global_step)
            if args.max_steps > 0 and global_step > args.max_steps:
                break
        # nb_train_steps = len(train_dataloader)
        results = {
            'loss': tr_loss / nb_train_steps,
            'loss_mlm': tr_loss_mlm / nb_train_steps,
            'loss_disc': tr_loss_disc / nb_train_steps,
            'acc_gen': tr_acc_gen / nb_train_steps,
            'acc_disc': tr_acc_disc / nb_train_steps,
        }
        output_train_file = os.path.join(args.output_dir, "train_results.txt")
        if os.path.exists(output_train_file):
            append_write = 'a+'  # append if already exists
        else:
            append_write = 'w'  # make a new file if not
        with open(output_train_file, append_write) as writer:
            logger.info("***** Train results {} *****".format(idx))
            writer.write('Train results %s\n' % idx)
            for key in sorted(results.keys()):
                logger.info("  %s = %s", key, str(results[key]))
                writer.write("%s = %s\n" % (key, str(results[key])))
        #         logger.info(" Training: Total Loss = %s, Generator loss = %s,Discriminator Loss = %s",tr_loss,tr_loss_mlm,tr_loss_disc)
        #         logger.info(" Training: Generator Accuracy = %s,Discriminator Accuracy = %s",tr_acc_gen,tr_acc_disc)

        if args.do_eval and (args.local_rank == -1 or torch.distributed.get_rank() == 0):
            results = evaluate(args, eval_dataset, model, tokenizer, pad_token_id, prefix=idx,
                               checkpoint=str(args.start_epoch + idx))

            last_output_dir = os.path.join(args.output_dir, 'checkpoint-last')
            if not os.path.exists(last_output_dir):
                os.makedirs(last_output_dir)
            generator_path = os.path.join(last_output_dir, 'generator')
            discriminator_path = os.path.join(last_output_dir, 'discriminator')
            gen_model_to_save = generator.module if hasattr(generator,
                                                            'module') else generator  # Take care of distributed/parallel training
            disc_model_to_save = discriminator.module if hasattr(discriminator,
                                                                 'module') else discriminator
            gen_model_to_save.roberta.save_pretrained(generator_path)
            disc_model_to_save.electra.save_pretrained(discriminator_path)
            logger.info("Saving model checkpoint to %s", last_output_dir)
            idx_file = os.path.join(last_output_dir, 'idx_file.txt')
            with open(idx_file, 'w', encoding='utf-8') as idxf:
                idxf.write(str(args.start_epoch + idx) + '\n')

            torch.save(optimizer.state_dict(), os.path.join(last_output_dir, "optimizer.pt"))
            torch.save(scheduler.state_dict(), os.path.join(last_output_dir, "scheduler.pt"))
            logger.info("Saving optimizer and scheduler states to %s", last_output_dir)

            step_file = os.path.join(last_output_dir, 'step_file.txt')
            with open(step_file, 'w', encoding='utf-8') as stepf:
                stepf.write(str(global_step) + '\n')

            if (results['acc_disc'] > best_acc):
                best_acc = results['acc_disc']
                output_dir = os.path.join(args.output_dir, 'checkpoint-best')
                if not os.path.exists(output_dir):
                    os.makedirs(output_dir)
                generator_path = os.path.join(output_dir, 'generator')
                discriminator_path = os.path.join(output_dir, 'discriminator')
                gen_model_to_save = generator.module if hasattr(generator,
                                                                'module') else generator  # Take care of distributed/parallel training
                disc_model_to_save = discriminator.module if hasattr(discriminator,
                                                                     'module') else discriminator
                gen_model_to_save.roberta.save_pretrained(generator_path)
                disc_model_to_save.electra.save_pretrained(discriminator_path)
                torch.save(args, os.path.join(output_dir, 'training_{}.bin'.format(idx)))
                logger.info("Saving model checkpoint to %s", output_dir)

                torch.save(optimizer.state_dict(), os.path.join(output_dir, "optimizer.pt"))
                torch.save(scheduler.state_dict(), os.path.join(output_dir, "scheduler.pt"))
                logger.info("Saving optimizer and scheduler states to %s", output_dir)

        if args.max_steps > 0 and global_step > args.max_steps:
            train_iterator.close()
            break

    if args.local_rank in [-1, 0]:
        tb_writer.close()

    return ""


def evaluate(args, eval_dataset, model, tokenizer, pad_token_id, checkpoint=None, prefix="", mode='dev'):
    args.eval_batch_size = args.per_gpu_eval_batch_size * max(1, args.n_gpu)
    # Note that DistributedSampler samples randomly
    eval_sampler = SequentialSampler(eval_dataset) if args.local_rank == -1 else DistributedSampler(eval_dataset)
    eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler, batch_size=args.eval_batch_size,
                                 collate_fn=lambda b: collate_batch(b, pad_token_id))

    # Eval!
    logger.info("***** Running evaluation {} *****".format(prefix))
    logger.info("  Num examples = %d", len(eval_dataset))
    logger.info("  Batch size = %d", args.eval_batch_size)
    tr_loss, tr_loss_mlm, tr_loss_disc, tr_acc_gen, tr_acc_disc = 0.0, 0.0, 0.0, 0.0, 0.0
    nb_eval_steps = 0
    preds = None
    out_label_ids = None
    for batch in tqdm(eval_dataloader, desc="Evaluating"):
        model.eval()
        batch = tuple(t.to(args.device) for t in batch)

        with torch.no_grad():
            loss, loss_mlm, loss_disc, acc_gen, acc_disc, disc_labels, disc_pred = model(batch[0],
                                                                                         attention_mask=batch[1])
            #                 tmp_eval_loss, logits = outputs[:2]
            loss = loss.mean()  # mean() to average on multi-gpu parallel training
            loss_mlm = loss_mlm.mean()
            loss_disc = loss_disc.mean()
            acc_gen = acc_gen.mean()
            acc_disc = acc_disc.mean()
            tr_loss += loss.item()
            tr_loss_mlm += loss_mlm.item()
            tr_loss_disc += loss_disc.item()
            tr_acc_gen += acc_gen.item()
            tr_acc_disc += acc_disc.item()

        nb_eval_steps += 1
    tr_loss /= nb_eval_steps
    tr_loss_mlm /= nb_eval_steps
    tr_loss_disc /= nb_eval_steps
    tr_acc_gen /= nb_eval_steps
    tr_acc_disc /= nb_eval_steps
    results = {
        'loss': tr_loss,
        'loss_mlm': tr_loss_mlm,
        'loss_disc': tr_loss_disc,
        'acc_gen': tr_acc_gen,
        'acc_disc': tr_acc_disc,
    }
    if (mode == 'dev'):
        output_eval_file = os.path.join(args.output_dir, "eval_results.txt")
        if os.path.exists(output_eval_file):
            append_write = 'a+'  # append if already exists
        else:
            append_write = 'w'  # make a new file if not
        with open(output_eval_file, append_write) as writer:
            logger.info("***** Eval results {} *****".format(prefix))
            writer.write('evaluate %s\n' % checkpoint)
            for key in sorted(results.keys()):
                logger.info("  %s = %s", key, str(results[key]))
                writer.write("%s = %s\n" % (key, str(results[key])))

    return results


def get_dataset(path_name, token_pad, tokenizer):
    data = pd.read_csv(path_name)
    logging.info(data.columns)
    tokens = tokenizer(data['combined'].values.tolist(), data['diff_tokens'].values.tolist(),
                       max_length=token_pad, padding=True, truncation=True)
    seq = torch.tensor(tokens['input_ids'])
    mask = torch.tensor(tokens['attention_mask'])
    tensor_data = TensorDataset(seq, mask)
    return tensor_data


def main():
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
    parser.add_argument('--seed', type=int, default=42,
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
    checkpoint_last = os.path.join(args.output_dir, 'checkpoint-last')
    if os.path.exists(checkpoint_last) and os.listdir(checkpoint_last):
        args.gen_model_name_or_path = os.path.join(checkpoint_last, 'generator')
        logger.info("Generator Last Checkpoint {}", args.gen_model_name_or_path)
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
    generator = RobertaForMaskedLM.from_pretrained(args.gen_model_name_or_path)
    for param in generator.roberta.parameters():
        param.requires_grad = False
    logger.info("Generator {}".format(generator.config._name_or_path))
    discriminator = ElectraForPreTraining(AutoConfig.from_pretrained(args.dis_model_name_or_path))
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
        train_dataset = get_dataset(args.train_file, args.max_seq_length, tokenizer)
        eval_dataset = get_dataset(args.dev_file, args.max_seq_length, tokenizer)
        if not os.path.exists(args.output_dir) and args.local_rank in [-1, 0]:
            os.makedirs(args.output_dir)
        global_step = train(args, train_dataset, eval_dataset, model, generator, discriminator, tokenizer, optimizer,
                            pad_token_id)
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
        gen_model_to_save.roberta.save_pretrained(generator_path)
        disc_model_to_save.electra.save_pretrained(discriminator_path)
        torch.save(args, os.path.join(args.output_dir, 'training_args.bin'))


if __name__ == "__main__":
    main()
