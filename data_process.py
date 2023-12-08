import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import re
import nltk
from pyvi import ViTokenizer
nltk.download('punkt')  
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
# read the file
df = pd.read_csv('data/crawl/synthetic_train.csv', encoding='utf-8  ')
# print(df.head())
# print(df.shape)
# print(df.info())
# preprocessing data
content = df.iloc[:, 0].values
# tokenize the words
#print content to see the data
def data_preprocessing(sentence):
    sentence = re.sub(r'[,!?;-]+', '.', str(sentence)) 
    sentence = sentence.lower()
    sentence = nltk.word_tokenize(sentence)
    return sentence

for sentence in content:
    sentence = data_preprocessing(sentence)
    print(sentence)
    
    