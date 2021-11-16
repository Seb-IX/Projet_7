from keras.models import load_model
import joblib

import tensorflow as tf

from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer
from keras.models import Sequential
from keras.layers import Dense, Embedding, LSTM, Dropout
from keras.wrappers.scikit_learn import KerasClassifier

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
import re


########### Class de la pipeline
class PadSequencesTransformer(BaseEstimator, TransformerMixin):

    def __init__(self, maxlen):
        self.maxlen = maxlen

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        X_padded = pad_sequences(X, maxlen=self.maxlen)
        return X_padded

    
class TokenizerTransformer(BaseEstimator, TransformerMixin, Tokenizer):

    def __init__(self, **tokenizer_params):
        Tokenizer.__init__(self, **tokenizer_params)

    def fit(self, X, y=None):
        self.fit_on_texts(X)
        return self

    def transform(self, X, y=None):
        X_transformed = self.texts_to_sequences(X)
        return X_transformed
    
def create_best_model(vocab_size=10000,padding=128,num_unit=1028):
    best_model = tf.keras.models.Sequential([
        tf.keras.layers.Embedding(vocab_size, padding,input_length = 40),
        tf.keras.layers.Dropout(0.1),
        tf.keras.layers.LSTM(num_unit),
        tf.keras.layers.Dropout(0.1),
        tf.keras.layers.Dense(2, activation="softmax"),
    ])
    best_model.compile(optimizer="adam",loss='categorical_crossentropy',metrics=['accuracy'])
    return best_model

########## Méthode de nettoyage

def clean_url(sentence):
    sentence = re.sub('http[s]*://[0-9a-zA-Z-_.]*.[a-z]{0,3}/[0-9a-zA-Z-_./]*', '', sentence)
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


def clean_escape(sentence):
    sentence = re.sub('&amp;','&',sentence)
    sentence = re.sub('&quot;','\"',sentence)
    sentence = re.sub('&gt;','>',sentence)
    sentence = re.sub('&lt;','<',sentence)
    return sentence

def clean_uniuque_char(sentence):
    sentence = " ".join([v for v in sentence.split(" ") if ((len(v) > 1) or (not v.isalpha()))])
    return sentence

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

def preprocessed_data(data : list):
    return [cleanning_data(tweet) for tweet in data]



######### Sauvegarde & récupération des données


def get_model_save(path_to_model: str="./data/model"):
    # Load the pipeline first:
    pipeline = joblib.load(path_to_model+"/pipeline_best_model.pkl")

    # Then, load the Keras model:
    pipeline.named_steps['model'].model = load_model(path_to_model+'/keras_model.h5')
    return pipeline


def create_pipeline():
    my_tokenizer = TokenizerTransformer(num_words=10000,split=' ')
    my_padder = PadSequencesTransformer(maxlen=40)
    my_model = KerasClassifier(build_fn=create_best_model, epochs=1)

    pipeline = Pipeline([
                  ('tokenizer', my_tokenizer),
                  ('padder', my_padder),
                  ('model', my_model)
    ])
    return pipeline
    
def save_pipeline_keras(pipeline, path_to_model_data: str = "./data/model"):
    # On sauvegarde a part pour évité les erreurs
    pipeline.named_steps['model'].model.save(path_to_model_data+'/keras_model.h5')

    # permet de sauvegarder correctement la pipeline
    pipeline.named_steps['model'].model = None

    joblib.dump(pipeline, path_to_model_data+"/pipeline_best_model.pkl")

