

def get_ngrams(token_list,n):
    temp = []
    for x in zip(*[token_list[i:] for i in range(n)]):
        temp.append(' '.join(x))
    return temp
