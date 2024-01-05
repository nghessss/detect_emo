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
df = pd.read_csv('data/crawl/synthetic_train.csv', encoding='utf-8  ')
content = df.iloc[:, 0].values
label = df.iloc[:, 1].values
def data_preprocessing(sentence):
    sentence = re.sub(r'[,!?;-]+', '.', str(sentence)) 
    sentence = sentence.lower()
    sentence = ViTokenizer.tokenize(sentence) # gộp các từ có nghĩa lại với nhau
    sentence = nltk.word_tokenize(sentence)
    return sentence


if __name__ == '__main__':
    #change directly to content
    for i in range(len(content)):
        content[i] = data_preprocessing(content[i])
    # for sentence in content:
    #     print(sentence)
    # init vocab
    vocab = train.init_vocab(content)
    # print(vocab)
    # change all data that out of vocabulary to <unk>
    n_gram = []
    for sentence in content:
        # for i in range(1,4):
        n_gram += train.n_gram(sentence, 2)
    #Count frequency of each n-gram
    print(n_gram)
    n_gram_freq = train.count_freq_n_gram(n_gram)
    for sentence, second_dimension in n_gram_freq.items():
        print(sentence, second_dimension)
        
    # lấy từng câu trong content ra để cộng trừ n-gram
    




   
