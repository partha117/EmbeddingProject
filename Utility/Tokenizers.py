import re


def whitespace_tokenizer(text):
    return [token for token in text.split()]


def underscore_tokenizer(text):
    return [token for token in text.split("_")]


def dot_tokenizer(text):
    return [token for token in text.split(".")]


def camelcase_tokenizer(text):
    matches = re.finditer('.+?(?:(?<=[a-z])(?=[A-Z])|(?<=[A-Z])(?=[A-Z][a-z])|$)', text)
    return [m.group(0) for m in matches]


def all_tokenizer(text):
    token_list = []
    for token1 in whitespace_tokenizer(text):
        for token2 in dot_tokenizer(token1):
            for token3 in underscore_tokenizer(token2):
                for token4 in camelcase_tokenizer(token3):
                    token_list.append(token4)
    return token_list
