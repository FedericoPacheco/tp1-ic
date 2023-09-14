from sklearn.model_selection import train_test_split
import pandas as pd
import spacy
import re
import emoji 
import contractions
import os

"""
pip install contractions
pip install emoji
pip install pandas
pip install spacy
python -m spacy download en
python -m spacy download en_core_web_md
python -m spacy download es_core_news_sm
"""

TRAINING_AND_VALIDATION_FILE = "twitter_training_and_validation.csv"
TESTING_FILE = "twitter_testing.csv"
CHOSEN_ENTITY = "Google"

TWEETS_FILE = "tweetsMilei.csv"

TRAINING_PREPROC = "training_preproc.csv"
VALIDATION_PREPROC = "validation_preproc.csv"
TESTING_PREPROC = "testing_preproc.csv"
DATASET = "dataset_completo.csv"


def get_raw_tweets_kaggle():
    # Cambiar al directorio del dataset 
    # os.chdir("..")
    os.chdir(os.path.join(os.getcwd(), "dataset/kaggle"))
    
    # Leer del csv asignando nombres a las columnas y descartando la primera (id)
    headers = ["id", "entity", "sentiment", "tweet"]
    training_and_validation_df = pd.read_csv(TRAINING_AND_VALIDATION_FILE, names = headers, usecols = headers[1:])
    testing_df = pd.read_csv(TESTING_FILE, names = headers, usecols = headers[1:])
    
    # Quedarse solo con la entidad de interes
    training_and_validation_df = training_and_validation_df[training_and_validation_df["entity"] == CHOSEN_ENTITY]
    testing_df = testing_df[testing_df["entity"] == CHOSEN_ENTITY]

    # Concatenacion de ambos datasets para luego hacer un split 70:10:20    
    dataset = pd.concat([training_and_validation_df, testing_df], ignore_index=True)    

    #return training_and_validation_df, testing_df
    return dataset


def get_raw_tweets():
    # Cambiar al directorio del dataset 
    # os.chdir("..")
    os.chdir(os.path.join(os.getcwd(), "dataset/tweets"))
    
    # Leer del csv asignando nombres a las columnas y descartando la primera (fecha)
    headers = ["fecha", "tweet", "sentiment"]
    dataset = pd.read_csv(TWEETS_FILE, usecols = headers[1:], sep=";")   

    return dataset

# Sacar cosas innecesarias del texto
def clean_text(raw_text):
    text = str(raw_text)                                # Convertir a cadena por si acaso
    text = contractions.fix(text)                       # Sacar abreviaturas
    text = emoji.demojize(text)                         # Sacar los emojis
    text = re.sub(r"(http|www)\S+", "", text)           # Remover urls
    text = text.lower()                                 # Llevar todo a minuscula
    text = re.sub(r"\d+", "", text)                     # Remover numeros
    text = re.sub(r"[\t\n\r\f\v]", "", text)            # Remover enters y otras "porquerias"
    text = re.sub(r"[\.\,:;]", " ", text)               # Remover caracteres de puntuacion innecesarios
    text = re.sub(r"[\[\]\(\)\{\}]", "", text)
    text = re.sub(r"[\"´`'′’“”<>]", "", text)
    text = re.sub(r"[¿\?¡\!\@_~\+\*°#%\|\-\$/&–—…]", "", text)   
    text = re.sub(r"\s{2,}", " ", text)                # Remover espacios de más   

    return text


def preprocess_tweets(df, idioma_ingles):
    
    modelo = ""
    if(idioma_ingles): modelo = "en_core_web_md"
    else : modelo = "es_core_news_sm"
    
    # Carga modelo para nlp (instalar previamente)
    nlp = spacy.load(modelo)
    
    # Operar "vectorialmente" sobre el data frame: limpiar texto, armar un doc de spacy con nlp(), fijarse que no sea stopword y lematizar
    df["tweet_preproc"] = df["tweet"].apply(
        lambda text: " ".join([token.lemma_ for token in nlp(clean_text(text)) if not token.is_stop])
    )
    
    # Sacar los nan y tweets vacios
    df = df[df["tweet_preproc"] != ""]
    df = df[df["tweet_preproc"] != " "]
    df = df.dropna(ignore_index=True)    
    
    return df
    
def get_test_train_validation(df):
    # Dividir el DataFrame en conjunto de entrenamiento (70%), conjunto de validación (20%) y conjunto de prueba (10%)
    train_df, temp_df = train_test_split(df, test_size=0.3)
    test_df, val_df = train_test_split(temp_df, test_size=0.67)
        
    return train_df, test_df, val_df


def save_dataset(df, name):
    new_file_name = "./preproc/" + name
    df[["tweet_preproc", "sentiment"]].to_csv(new_file_name, index = False)


def procesarDatosKaggle():

    dataset = get_raw_tweets_kaggle() # kaggle

    dataset_preproc = preprocess_tweets(dataset, idioma_ingles=True) # idioma_ingles=True
    
    train_df, test_df, val_df = get_test_train_validation(dataset_preproc)
    
    save_dataset(dataset_preproc, DATASET)
    save_dataset(train_df, TRAINING_PREPROC)
    save_dataset(test_df, TESTING_PREPROC)
    save_dataset(val_df, VALIDATION_PREPROC)


def procesarDatosTweets():

    dataset = get_raw_tweets() # tweets

    dataset_preproc = preprocess_tweets(dataset, idioma_ingles=False) # idioma_ingles=False
    
    train_df, test_df, val_df = get_test_train_validation(dataset_preproc) 
    
    save_dataset(dataset_preproc, DATASET)
    save_dataset(train_df, TRAINING_PREPROC)
    save_dataset(test_df, TESTING_PREPROC)
    save_dataset(val_df, VALIDATION_PREPROC)


if __name__ == "__main__":
    
    procesarDatosKaggle()
    # procesarDatosTweets()
    
    


