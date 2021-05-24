import torch.nn as nn
import torch
from functools import reduce
import logging
import os
import random
import math
import torch.nn.functional as F
from tqdm import tqdm, trange
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, TensorDataset
from tensorboardX import SummaryWriter
from transformers import RobertaModel, BertPreTrainedModel, AutoConfig, \
    get_linear_schedule_with_warmup, AdamW, AutoModel
from transformers.models.electra.modeling_electra import ElectraDiscriminatorPredictions
import numpy as np


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


class ElectraForPreTraining(BertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.electra = AutoModel.from_config(config)
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
            output_hidden_states=True,
            return_dict=None,
    ):
        # discriminator_hidden_states = self.electra(
        #     input_ids,
        #     attention_mask,
        #     token_type_ids,
        #     position_ids,
        #     head_mask,
        #     inputs_embeds,
        #     output_attentions,
        #     output_hidden_states,
        #     return_dict,
        # )
        # #print("discriminator_hidden_states", discriminator_hidden_states.shape)
        # discriminator_sequence_output = discriminator_hidden_states[0]
        discriminator_outputs = self.electra(
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
        #print("discriminator_hidden_states", discriminator_outputs, output_hidden_states)
        discriminator_sequence_output = discriminator_outputs.hidden_states[0]
        logits = self.discriminator_predictions(discriminator_sequence_output)
        return logits


def tie_weights(generator, discriminator):
    # generator.roberta.embeddings.word_embeddings = discriminator.electra.embeddings.word_embeddings
    # generator.roberta.embeddings.position_embeddings = discriminator.electra.embeddings.position_embeddings
    # generator.roberta.embeddings.token_type_embeddings = discriminator.electra.embeddings.token_type_embeddings
    discriminator.electra.embeddings.word_embeddings = generator.reformer.embeddings.word_embeddings
    discriminator.electra.embeddings.position_embeddings = generator.reformer.embeddings.position_embeddings
    #discriminator.electra.embeddings.token_type_embeddings = generator.reformer.embeddings.token_type_embeddings


class LogitsAdapter(torch.nn.Module):
    def __init__(self, adaptee):
        super().__init__()
        self.adaptee = adaptee

    def forward(self, *args, **kwargs):
        return self.adaptee(*args, **kwargs)[0]


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
            #print("Random", torch.max(random_indices),torch.min(random_indices))
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


def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)


def collate_batch(examples, pad_token_id):
    # print(examples[0][0])
    input_ids = torch.nn.utils.rnn.pad_sequence([torch.tensor(example['input_ids']) for example in examples],
                                                batch_first=True,
                                                padding_value=pad_token_id)
    input_mask = torch.nn.utils.rnn.pad_sequence([torch.tensor(example['attention_mask']) for example in examples],
                                                 batch_first=True,
                                                 padding_value=pad_token_id)
    return input_ids, input_mask


def return_sorted(path, function, return_last=False, return_first=False):
    name_store = dict()
    for item in os.listdir(path):
        if function(item):
            name_store[function(item)] = item
    name_store = dict(sorted(name_store.items()))
    if len(name_store.keys()) > 0:
        if return_last:
            return name_store[list(name_store.keys())[-1]], list(name_store.keys())[-1]
        elif return_first:
            return name_store[list(name_store.keys())[0]], list(name_store.keys())[0]
        return name_store
    else:
        return None, 0


def train(args, train_dataset, eval_dataset, model, generator, discriminator, tokenizer, optimizer, pad_token_id,
          logger):
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
                logger.info(" Training: Total Loss = %s, Generator loss = %s,Discriminator Loss = %s", tr_loss,
                            tr_loss_mlm, tr_loss_disc)
                logger.info(" Training: Generator Accuracy = %s,Discriminator Accuracy = %s", tr_acc_gen, tr_acc_disc)

        if args.max_steps > 0 and global_step > args.max_steps:
            train_iterator.close()
            break

        last_output_dir = os.path.join(args.output_dir, 'checkpoint-{}'.format(idx))
        if not os.path.exists(last_output_dir):
            os.makedirs(last_output_dir)
        generator_path = os.path.join(last_output_dir, 'generator')
        discriminator_path = os.path.join(last_output_dir, 'discriminator')
        gen_model_to_save = generator.module if hasattr(generator,
                                                        'module') else generator  # Take care of distributed/parallel training
        disc_model_to_save = discriminator.module if hasattr(discriminator,
                                                             'module') else discriminator
        gen_model_to_save.reformer.save_pretrained(generator_path)
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
    if args.local_rank in [-1, 0]:
        tb_writer.close()

    return ""
