from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd
TF_IDF= TfidfVectorizer()
df = pd.read_csv('datasets/Fixed_Jobs_Dataset.csv', encoding="utf-8", delimiter=";", on_bad_lines="skip")
print(df.head(5))