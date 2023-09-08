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
"""

TRAINING_AND_VALIDATION_FILE = "twitter_training.csv"
TESTING_FILE = "twitter_validation.csv"
CHOSEN_ENTITY = "Google"


def get_dataset():
    # Cambiar al directorio del dataset 
    os.chdir("..")
    os.chdir(os.path.join(os.getcwd(), "dataset"))
    
    # Leer del csv asignando nombres a las columnas y descartando la primera (id)
    headers = ["id", "entity", "sentiment", "tweet"]
    training_and_validation_df = pd.read_csv(TRAINING_AND_VALIDATION_FILE, names = headers, usecols = headers[1:])
    testing_df = pd.read_csv(TESTING_FILE, names = headers, usecols = headers[1:])
    
    # Quedarse solo con la entidad de interes
    training_and_validation_df = training_and_validation_df[training_and_validation_df["entity"] == CHOSEN_ENTITY]
    testing_df = testing_df[testing_df["entity"] == CHOSEN_ENTITY]

    return training_and_validation_df, testing_df


def preprocess_tweets(df):
    # Carga modelo para nlp (instalar previamente)
    nlp = spacy.load("en_core_web_md")
    
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
    
    # Operar "vectorialmente" sobre el data frame: limpiar texto, armar un doc de spacy con nlp(), fijarse que no sea stopword y lematizar
    df["tweet_preproc"] = df["tweet"].apply(
        lambda text: " ".join([token.lemma_ for token in nlp(clean_text(text)) if not token.is_stop])
    )
    
    return df
    

def save_to_disk(df, file_name):
    new_file_name = file_name.rsplit(".", 1)[0] + "-preproc.csv"
    df[["sentiment", "tweet_preproc"]].to_csv(new_file_name, index = False)


def main():
    training_and_validation_df, testing_df = get_dataset()

    preprocess_tweets(training_and_validation_df)
    preprocess_tweets(testing_df)

    save_to_disk(training_and_validation_df, TRAINING_AND_VALIDATION_FILE)
    save_to_disk(testing_df, TESTING_FILE)

    #print(training_and_validation_df)
    #print(testing_df)


if __name__ == "__main__":
    main()


