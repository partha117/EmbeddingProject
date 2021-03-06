from collections import Counter
from torchtext.vocab import Vocab

from Utils.utility import get_ngrams
import pickle
import copy
import os
from pathlib import Path


def create_title_description_vocab(title_data, description_data, pipeline, ngram=1, unk='<unk>', pad='<pad>',
                                   minfreq=1):
    vocab_counter = Counter()
    for i, item in enumerate(title_data):
        vocab_counter.update(get_ngrams(pipeline(item), ngram))
        print("\r Title completed: {}% ".format(round((i / len(title_data)) * 100, 2)), end='', flush=True)
    print("\n")
    for i, item in enumerate(description_data):
        vocab_counter.update(get_ngrams(pipeline(item), ngram))
        print("\r Description completed: {}% ".format(round((i / len(description_data)) * 100, 2)), end='', flush=True)
    vocab = Vocab(vocab_counter, specials=[unk, pad], min_freq=minfreq)
    return vocab


def create_source_code_vocab(after_fix_code_data, pipeline, ngram=1, unk='<unk>', pad='<pad>',
                             min_freq=1):
    vocab_counter = Counter()
    for i, item in enumerate(after_fix_code_data):
        with open(item, "r") as file:
            file_content = file.read()
            vocab_counter.update(get_ngrams(pipeline(file_content), ngram))
            print("\r After fix code completed: {}% ".format(round((i / len(after_fix_code_data)) * 100, 2)), end='',
                  flush=True)
    vocab = Vocab(vocab_counter, specials=[unk, pad], min_freq=min_freq)
    return vocab


def get_title_description_vocabulary(title_data, description_data, pipeline, ngram=1, unk='<unk>', pad='<pad>',
                                     min_freq=1,
                                     fresh=False):
    if not fresh and os.path.isfile(os.path.abspath("Output/title_description_vocab.pt")):
        return pickle.load(open(os.path.abspath("Output/title_description_vocab.pt"), "rb"))
    else:
        Path("Output/").mkdir(parents=True, exist_ok=True)
        vocab = create_title_description_vocab(title_data, description_data, pipeline, ngram, unk, pad, min_freq)
        pickle.dump(vocab, open(os.path.abspath("Output/title_description_vocab.pt"), "wb"))
        return vocab


def get_source_code_vocabulary(after_fix_code_data, pipeline, ngram=1, unk='<unk>', pad='<pad>',
                               min_freq=1,
                               fresh=False):
    if not fresh and os.path.isfile(os.path.abspath("Output/source_code_vocab.pt")):
        return pickle.load(open(os.path.abspath("Output/source_code_vocab.pt"), "rb"))
    else:
        Path("Output/").mkdir(parents=True, exist_ok=True)
        vocab = create_source_code_vocab(after_fix_code_data, pipeline, ngram, unk, pad,
                                         min_freq=min_freq)
        pickle.dump(vocab, open(os.path.abspath("Output/source_code_vocab.pt"), "wb"))
        return vocab


def join_vocab(vocabulary_one, vocabulary_two):
    joined_vocab = copy.deepcopy(vocabulary_one)
    joined_vocab.extend(vocabulary_two)
    return joined_vocab


def test_vocabulary(vocabulary, pipeline, text):
    string_to_id = [vocabulary.stoi[t] for t in pipeline(text)]
    print(string_to_id)
    id_to_string = [vocabulary.itos[t] for t in string_to_id]
    print(id_to_string)


def get_title_description_corpus(title_data, description_data, pipeline, fresh=False):
    if not fresh and os.path.isfile(os.path.abspath("Output/title_description_corpus.pt")):
        return pickle.load(open(os.path.abspath("Output/title_description_corpus.pt"), "rb"))
    else:
        title_description_corpus = []
        for i, item in enumerate(title_data):
            title_description_corpus.append(pipeline(item))
            print("\r Title completed: {}% ".format(round((i / len(title_data)) * 100, 2)), end='', flush=True)
        print("\n")
        for i, item in enumerate(description_data):
            title_description_corpus.append(pipeline(item))
            print("\r Description completed: {}% ".format(round((i / len(description_data)) * 100, 2)), end='',
                  flush=True)
        pickle.dump(title_description_corpus, open(os.path.abspath("Output/title_description_corpus.pt"), "wb"))
        return title_description_corpus


def get_source_code_corpus(before_fix_code_data, after_fix_code_data, pipeline, fresh=False):
    if not fresh and os.path.isfile(os.path.abspath("Output/source_code_corpus.pt")):
        return pickle.load(open(os.path.abspath("Output/source_code_corpus.pt"), "rb"))
    else:
        source_code_corpus = []
        for i, item in enumerate(before_fix_code_data):
            with open(item, "r") as file:
                file_content = file.read()
                source_code_corpus.append(pipeline(file_content))
                print("\r Before fix code completed: {}% ".format(round((i / len(before_fix_code_data)) * 100, 2)),
                      end='', flush=True)
        print("\n")
        for i, item in enumerate(after_fix_code_data):
            with open(item, "r") as file:
                file_content = file.read()
                source_code_corpus.append(pipeline(file_content))
                print("\r After fix code completed: {}% ".format(round((i / len(after_fix_code_data)) * 100, 2)),
                      end='', flush=True)
        pickle.dump(source_code_corpus, open(os.path.abspath("Output/source_code_corpus.pt"), "wb"))
        return source_code_corpus


def get_title_description_iterator(title_data, description_data, pipeline):
    class TitleDescriptionIterator(object):
        def __init__(self, title_data, description_data, pipeline):
            self.title_data = list(title_data)
            self.description_data = list(description_data)
            self.pipeline = pipeline
            self.current_position = 0
            self.title_data_status = True
            self.description_data_status = False

        def __iter__(self):
            return self

        def __len__(self):
            return len(self.title_data) + len(self.description_data)

        def __next__(self):
            if self.title_data_status:
                if len(self.title_data) > self.current_position:
                    self.current_position += 1
                    return self.pipeline(self.title_data[self.current_position - 1])
                else:
                    self.current_position = 0
                    self.title_data_status = False
                    self.description_data_status = True
            if self.description_data_status:
                if len(self.description_data) > self.current_position:
                    self.current_position += 1
                    return self.pipeline(self.description_data[self.current_position - 1])
                else:
                    self.current_position = 0
                    self.title_data_status = True
                    self.description_data_status = False
                    raise StopIteration

    return TitleDescriptionIterator(title_data, description_data, pipeline)


def get_source_code_iterator(after_fix_code_data, pipeline):
    class SourceCodeIterator(object):
        def __init__(self, after_fix_code_data, pipeline):
            self.after_fix_code_data = list(after_fix_code_data)
            self.pipeline = pipeline
            self.current_position = 0

        def __iter__(self):
            return self

        def __len__(self):
            return len(self.after_fix_code_data)

        def __next__(self):
            if len(self.after_fix_code_data) > self.current_position:
                self.current_position += 1
                with open(self.after_fix_code_data[self.current_position - 1], "r") as file:
                    file_content = file.read()
                    print("\r After Source code completed: {}% ".format(
                        round((self.current_position / len(self.after_fix_code_data)) * 100, 2)), end='',
                        flush=True)
                    return self.pipeline(file_content)
            else:
                self.current_position = 0
                raise StopIteration

    return SourceCodeIterator(after_fix_code_data, pipeline)
