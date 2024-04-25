import pandas as pd
import pickle
import nltk
from nltk.corpus import stopwords
nltk.download('stopwords') 
from pymorphy2 import MorphAnalyzer
import re

patterns = "[A-Za-z0-9!#$%&'()*+,./:;<=>?@[\]^_`{|}~—\"\-]+"
stopwords_ru = stopwords.words("russian")  #Стоп-слвоа
morph = MorphAnalyzer()
def lemmatize(doc):
    doc = re.sub(patterns, ' ', doc)
    tokens = []
    for token in doc.split():
        if token and token not in stopwords_ru:
            token = token.strip()
            token = morph.normal_forms(token)[0]
            
            tokens.append(token)
    
    return ' '.join(tokens)

def get_result(text: pd.Series) -> pd.Series:
    loaded_model = pickle.load(open('Mtex_clf', "rb"))
    
   return pd.DataFrame({'class_predicted': loaded_model.predict(text.apply(lemmatize))})
 

