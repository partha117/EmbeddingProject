import re


def convert_to_lower(text):
    return text.lower()


def non_english_filter(text):
    non_ascii_regex = r"[^\x00-\x7F]+"
    return re.sub(non_ascii_regex, "", text)


def char_filter(text):
    filters = '!"#$%&()*+,-/:;<=>?@[\\]^_`{|}~'
    regex = ''
    for item in filters:
        regex = regex + "\\" + item + "|"
    return re.sub(regex, "", text)


def digit_filter(text):
    digit_regex = r"(\s|^)\d+\.{0,1}\d*(\s|$)"
    #     digit_regex = r"\d"
    return re.sub(digit_regex, "", text)


def url_filter(text):
    # url_regex = r"(?i)\b((?:https?://|www\d{0,3}[.]|[a-z0-9.\-]+[.][a-z]{2,4}/)(?:[^\s()<>]+|\(([^\s()<>]+|(\([^\s()<>]+\)))*\))+(?:\(([^\s()<>]+|(\([^\s()<>]+\)))*\)|[^\s`!()\[\]{};:'\".,<>?«»“”‘’]))"
    url_regex = r"(http|https|www)(:|\.)[^\s]+(\s|$)"
    return re.sub(url_regex, "", text)


def comment_filter(text):
    comment_regex = r"(/\*(.|\s)*\*/)|(\#.*\n)|(\"\"\"[\w\W]*?\"\"\")|(//.*\n)"
    return re.sub(comment_regex, "", text)


def dot_filter(text):
    dot_regex = "\\."
    return re.sub(dot_regex, "", text)
