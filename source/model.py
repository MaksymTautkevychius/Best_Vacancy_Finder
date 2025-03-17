from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd

def check_similarity(TF_IDF : TfidfVectorizer, Cosine_Sim : any, data : str)->str:
    matrix = TF_IDF.fit_transform(data)
    print(TF_IDF.vocabulary_)
    return