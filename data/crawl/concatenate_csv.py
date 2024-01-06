import pandas as pd
import matplotlib.pyplot as plt

df1 = pd.read_csv('./test.csv', encoding='utf-8  ')
df2 = pd.read_csv('./synthetic_val.csv', encoding='utf-8  ')
df = pd.concat([df1, df2])
df.to_csv('test.csv', index=False, encoding='utf-8')