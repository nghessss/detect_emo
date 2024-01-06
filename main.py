import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import re
import nltk
from pyvi import ViTokenizer
nltk.download('punkt')  
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
import train
# read the file

def data_preprocessing(sentence):
    sentence = re.sub(r'[,!?;-]+', '.', str(sentence)) 
    sentence = sentence.lower()
    sentence = ViTokenizer.tokenize(sentence) # gộp các từ có nghĩa lại với nhau
    sentence = nltk.word_tokenize(sentence)
    return sentence


if __name__ == '__main__':
    df = pd.read_csv('data/train.csv', encoding='utf-8  ')
    content_train = df.iloc[:, 0].values
    label_train = df.iloc[:, 1].values
    for i in range(len(content_train)):
        content_train[i] = data_preprocessing(content_train[i])
    
    vocab = train.init_vocab(content_train)

    # train
    n_gram = []
    for sentence in content_train:
        for i in range(1,4):
            n_gram += train.n_gram(sentence, i)
    n_gram_freq = train.count_freq_n_gram(n_gram)
    n_gram_freq = train.count_emotion(n_gram_freq,content_train, label_train)
    # print positive, negative, neutral in n_gram_freq
    # print(n_gram_freq)
    # for key, value in n_gram_freq.items():
    #     print(value)


    # predict
    df = pd.read_csv('data/test.csv', encoding='utf-8  ')
    content_val = df.iloc[:, 0].values
    label_val = df.iloc[:, 1].values
    for i in range(len(content_val)):
        content_val[i] = data_preprocessing(content_val[i])
    # print(content_val)
    # print(label_val)
    count = 0
    for i in range(len(content_val)):
        if (label_val[i] == 'neutral'):
            continue
        emo = train.predict_emo(n_gram_freq, content_val[i])
        # print(emo, label_val[i])
        if emo == label_val[i]:
            count += 1
    # sum without neutral
    sum_without_neutral = sum(label != 'neutral' for label in label_val ) 
    sum = len(label_val)
    print(sum_without_neutral)
    print(count/sum_without_neutral * 100, '%')
    
    



   
