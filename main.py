from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
from source.model import check_similarity
from source.data_preprocessing import read_text_from_image

def check_similarity(TF_IDF: TfidfVectorizer, Cosine_Sim: any, datapdf: list,vacancies) -> str:
    matrix1 = TF_IDF.fit_transform(data)  
    matrix2 = TF_IDF.fit_transform(vacancies)
    Cosine_Sim = cosine_similarity(matrix1, matrix2)  
    print("Vocabulary:", TF_IDF.vocabulary_)
    print("Cosine Similarity Matrix:", Cosine_Sim)
    

TF_IDF= TfidfVectorizer()
#check_similarity()
Csin=0
data=read_text_from_image()
vacancies  = pd.read_csv("datasets/Fixed_Jobs_Dataset.csv", encoding="utf-8", delimiter=";", on_bad_lines="skip")
print(vacancies.head(5))
vacancies = vacancies.drop(columns=['Label'],axis=1)
print(vacancies.head(5))

#check_similarity(TF_IDF,Csin,data,vacancies)
#df = pd.read_csv('datasets/Fixed_Jobs_Dataset.csv', encoding="utf-8", delimiter=";", on_bad_lines="skip")

#check_similarity(TF_IDF)