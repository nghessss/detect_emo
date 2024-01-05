import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# change all data that out of vocabulary to <unk>
def change_oov_to_unk(sentence, vocab):
    for i in range(len(sentence)):
        if sentence[i] not in vocab:
            sentence[i] = '<unk>'
    return sentence

def n_gram(sentence, n):
    n_gram = []
    # add to front and back of sentence n-1 <s> and </s> tokens
    sentence = ['<s>'] * (n-1) + sentence + ['</s>'] * (n-1)
    
    for i in range(len(sentence) - n + 1):
        n_gram.append(sentence[i:i+n])
    return n_gram

# get through all sentences to init vocab
def init_vocab(content):
    vocab = {}
    for sentence in content:
        for word in sentence:
            if word not in vocab:
                vocab[word] = 1
            else:
                vocab[word] += 1
    return vocab
# count frequency of each n-gram
def count_freq_n_gram(n_gram):
    n_gram_freq = {}
    second_dimension = {
        'positive': 0,
        'negative': 0,
        'neutral': 0,
        'frequency': 1
    }
    # each n-gram has a value is second_dimension count frequency of each n-gram
    for sentence in n_gram:
        # print(sentence)
        word = tuple(sentence)
        # print(word)
        if word not in n_gram_freq:
            n_gram_freq[word] = second_dimension.copy()
        else:
            n_gram_freq[word]['frequency'] += 1

    return n_gram_freq
# lấy từng câu trong content ra để cộng trừ n-gram
