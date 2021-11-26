import joblib
import json
import numpy as np
import os
import re

import tensorflow as tf

from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer
from keras.models import Sequential, load_model
from keras.layers import Dense, Embedding, LSTM, Dropout
from keras.wrappers.scikit_learn import KerasClassifier

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline

from azureml.core.model import Model


def init():
    global model
    global tokenise

    tokenizer_path = Model.get_model_path(model_name = 'tokenizer_keras')
    keras_path = Model.get_model_path(model_name = 'keras_w2v_model')

    try:
        tokenise = joblib.load(tokenizer_path)
    except:
        print("tokenizer not work")
        
    try: 
        model = load_model(keras_path)
    except:
        print("Keras model not working")

def run(request):
    try:
        req = json.loads(request)
        data = req["data"]
        method = req["method"] 
        if type(data) is list:
            data = preprocess_data(data)
        else:
            return "Please enter list of tweet as type of \"list\""
            
        if str(method) == "prediction":
            dic = {1:"good",0:"bad"}
            result = np.argmax(model.predict(data),axis=1)
            return [dic[y] for y in result]
            
        elif str(method) == "score":
            result = model.predict(data)
            return result[:,1].tolist()
        
        elif str(method) == "prediction_with_neutral":
            result = model.predict(data)
            result = result[:,1]
            return [predict_with_neutral(y) for y in result]
        else:
            return """method aviable : \"prediction\",\"score\",\"prediction_with_neutral\"
 - prediction return labels sentiment (bad/good)
 - score return score of sentiment 0(bad) to 1(good)
 - prediction_with_neutral return labels sentiment with label neutral"""
            
    except Exception as e:
        error = str(e)
        return error
    
def predict_with_neutral(score):
    if score >= 0.6:
        return "good"
    elif score <= 0.4:
        return "bad"
    else:
        return "neutral"

def preprocess_data(data : list):
    return pad_sequences(tokenise.texts_to_sequences([cleanning_data(tweet) for tweet in data]),maxlen=300)

def cleanning_data(sentence):
    sentence += " "
    sentence = re.sub('\n+', ' ', sentence.lower())
    # special caractère
    sentence = re.sub('[éèêë]','e',sentence)
    sentence = re.sub('[ïîì]','i',sentence)
    sentence = re.sub('[àâä]','a',sentence)
    sentence = re.sub('[ôö]','o',sentence) 
    sentence = re.sub('[üûù]','u',sentence)
    sentence = re.sub('[ç]','c',sentence)
    # personal_clean
    sentence = clean_escape(sentence) # usless ?
    sentence = clean_url(sentence)
    sentence = clean_smile(sentence)
    sentence = re.sub(r'[^a-z@#\'_ ]+',' ',sentence)
    sentence = clean_uniuque_char(sentence)
    sentence = re.sub(' {2,}', ' ', sentence)
    return sentence

def clean_uniuque_char(sentence):
    sentence = " ".join([v for v in sentence.split(" ") if ((len(v) > 1) or (not v.isalpha()))])
    return sentence

def clean_escape(sentence):
    sentence = re.sub('&amp;','&',sentence)
    sentence = re.sub('&quot;','\"',sentence)
    sentence = re.sub('&gt;','>',sentence)
    sentence = re.sub('&lt;','<',sentence)
    return sentence

def clean_smile(sentence):
    sentence = re.sub(';\)','emote_clin_doeil ',sentence)
    sentence = re.sub(':\)|=\)','emote_sourire ',sentence)
    sentence = re.sub('x\)','emote_cross_smile ',sentence)
    sentence = re.sub('x{1,}[Dd]{1,}[^A-Za-z0-9]','emote_mdr ',sentence)
    sentence = re.sub(';[pP]{1,}[^A-Za-z0-9]|:[pP]{1,}[^A-Za-z0-9]|x[pP]{1,}[^A-Za-z0-9]|=[pP]{1,}[^A-Za-z0-9]',
                      'emote_tire_langue ',sentence)
    sentence = re.sub('[oO]_[oO]','emote_tres_surpris',sentence)
    sentence = re.sub('[Xx]{1,}[Oo]{1,}[^A-Za-z0-9]','emote_ko ',sentence)
    sentence = re.sub('[bB]\)[^A-Za-z0-9]','emote_lunette ',sentence)
    sentence = re.sub('[=:;][oO]{1,}[^A-Za-z0-9]','emote_surpris ',sentence)
    sentence = re.sub('[=:;][S]{1,}[^A-Za-z0-9]','emote_genant ',sentence)
    sentence = re.sub('[:;=][dD]{1,}[^A-Za-z0-9]','emote_gros_sourire ',sentence)
    sentence = re.sub('[:;=][$]{1,}[^A-Za-z0-9]','emote_genant_discret ',sentence)
    sentence = re.sub('[:;=]\|[^A-Za-z0-9]','emote_neutre ',sentence)
    sentence = re.sub('[:;=]/[^A-Za-z0-9]','emote_deception ',sentence)
    sentence = re.sub('[:;=]\([^A-Za-z0-9]','emote_insatisfait ',sentence)
    sentence = re.sub('\*[-_]\*[^A-Za-z0-9]','emote_magnifique ',sentence)
    sentence = re.sub('\^[-_]{0,1}\^[^A-Za-z0-9]','emote_joyeux ',sentence)
    sentence = re.sub('--\'[^A-Za-z0-9]','emote_enerve ',sentence)
    return sentence

def clean_url(sentence):
    sentence = re.sub('http[s]*://[0-9a-zA-Z-_.]*.[a-z]{0,3}/[0-9a-zA-Z-_./]*', '', sentence)
    return sentence
