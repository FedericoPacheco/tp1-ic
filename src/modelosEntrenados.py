import pandas as pd
import tensorflow as tf
import os
from sklearn.preprocessing import LabelEncoder
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
import scipy

TRAINING_PREPROC = "training_preproc.csv"
VALIDATION_PREPROC = "validation_preproc.csv"
TESTING_PREPROC = "testing_preproc.csv"

TFIDF_BALOTAJE = "modelo-TFIDF-balotaje.keras"
EMBEDDING_BALOTAJE = "modelo-EMBEDDING-balotaje.keras"
CONVOLUTIONAL_BALOTAJE = "modelo-CONVOLUTIONAL-balotaje.keras"

def get_dataset(fuente): # "kaggle" o "tweets" o "balotaje2015"
    # Cambiar al directorio del dataset
    # os.chdir("..")
    os.chdir(os.path.join(os.getcwd(), "dataset/" + fuente + "/preproc"))

    # Leer del csv
    train_df = pd.read_csv(TRAINING_PREPROC)
    validation_df = pd.read_csv(VALIDATION_PREPROC)
    test_df = pd.read_csv(TESTING_PREPROC)
    
    return train_df, validation_df, test_df

def encodeLabel(training_df, validation_df, testing_df):

    # Converting label sentiment to numeric value
    training_df['encoded_sentiment'] = LabelEncoder(
    ).fit_transform(training_df["sentiment"])
    validation_df['encoded_sentiment'] = LabelEncoder(
    ).fit_transform(validation_df["sentiment"])
    testing_df['encoded_sentiment'] = LabelEncoder(
    ).fit_transform(testing_df["sentiment"])

    # Separar tweets preprocesados y labels codificados en Series de entrenaminento, validacion y testeo
    x_train = training_df["tweet_preproc"]
    y_train = training_df["encoded_sentiment"]

    x_val = validation_df["tweet_preproc"]
    y_val = validation_df["encoded_sentiment"]

    x_test = testing_df["tweet_preproc"]
    y_test = testing_df["encoded_sentiment"]

    return x_train, y_train, x_val, y_val, x_test, y_test


def modeloTFIDFBalotaje():
    
    os.chdir(os.path.join(os.getcwd(), "src/trainedModels/"))
    model = tf.keras.models.load_model(TFIDF_BALOTAJE)
    model.summary()
    
    training_df, validation_df, testing_df = get_dataset("tweets")
    x_train, y_train, x_val, y_val, x_test, y_test = encodeLabel(training_df, validation_df, testing_df)
    
    

def modeloEMBEDDINGFBalotaje():
    
    os.chdir(os.path.join(os.getcwd(), "src/trainedModels/"))
    model = tf.keras.models.load_model(EMBEDDING_BALOTAJE)
    model.summary()

    
def modeloCONVOLUTIONALBalotaje():
    
    os.chdir(os.path.join(os.getcwd(), "src/trainedModels/"))
    model = tf.keras.models.load_model(CONVOLUTIONAL_BALOTAJE)
    model.summary()



if __name__ == "__main__":
    modeloTFIDFBalotaje()
    modeloCONVOLUTIONALBalotaje()
    modeloEMBEDDINGFBalotaje()