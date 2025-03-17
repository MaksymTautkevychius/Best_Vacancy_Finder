from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
import numpy as np
from source.model import check_similarity
from source.data_preprocessing import read_text_from_image

def check_similarity(TF_IDF: TfidfVectorizer,Cosin_sim:None, datapdf: list, vacancies: pd.DataFrame) -> dict:
    
    matrix1 = TF_IDF.transform(datapdf)
    matrix2 = TF_IDF.transform(vacancies['Description'])
    

    
    cosine_similarities = {}

    
    for index in range(matrix2.shape[0]):  
        cosine_similarities[index] = cosine_similarity(matrix1, matrix2[index])
        
    best_fitted=0
    best_fitted_index=0
    for key, value in cosine_similarities.items():
        if(value>=best_fitted): 
            best_fitted=value
            best_fitted_index=key
    return best_fitted_index

TF_IDF= TfidfVectorizer()

data=read_text_from_image()
data=[data]
vacancies  = pd.read_csv("datasets/Fixed_Jobs_Dataset.csv", encoding="utf-8", delimiter=";", on_bad_lines="skip")
cosine = None
TF_IDF.fit(data + vacancies['Description'].tolist())
index=check_similarity(TF_IDF,cosine, data,vacancies)
print(f'best position is: {vacancies['Name of Position'][index]}')
print(f'link to job: {vacancies['Link to Job'][index]}')
print(f'Description: {vacancies['Description'][index]}')
