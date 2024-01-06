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
# count the emotion on the sentences
def count_emotion(n_gram_freq,content, label):
    for i in range(len(content)):
        sentence_n_gram = []
        for j in range(1,4):
            sentence_n_gram += n_gram(content[i], j)
        for sentence in sentence_n_gram:
            # print(sentence)
            word = tuple(sentence)
            # print(word)
            if (word not in n_gram_freq):
                continue
            if label[i] == 'positive':
                n_gram_freq[word]['positive'] += 1
            elif label[i] == 'negative':
                n_gram_freq[word]['negative'] += 1
            else:
                n_gram_freq[word]['neutral'] += 1
    return n_gram_freq
# lấy từng câu trong content ra để cộng trừ n-gram
def predict_emo(n_gram_freq, sentence):
    # log_prob contains negative, positive, neutral
    log_pro = {
        'positive': 0,
        'negative': 0,
        'neutral': 0
    }
    lamb = [0, 0.05, 0.1, 0.8]
    for i in range(1,4):
        sentence_n_gram = []
        sentence_n_gram = n_gram(sentence, i)
        for sentence in sentence_n_gram:
            word = tuple(sentence)  
            if (word not in n_gram_freq):
                continue
            log_pro['positive'] += np.log((n_gram_freq[word]['positive'] + 1) / (n_gram_freq[word]['frequency'] + len(n_gram_freq))*lamb[i])
            log_pro['negative'] += np.log((n_gram_freq[word]['negative'] + 1) / (n_gram_freq[word]['frequency'] + len(n_gram_freq))*lamb[i])
            log_pro['neutral'] += np.log((n_gram_freq[word]['neutral'] + 1) / (n_gram_freq[word]['frequency'] + len(n_gram_freq))*lamb[i])
            
    # log_pro['positive'] *= -1
    # log_pro['negative'] *= -1
    # log_pro['neutral'] *= -1
    print(log_pro)
    if log_pro['positive'] >= log_pro['negative']:
        return 'positive'
    else:
        return 'negative'
  