import pandas as pd
import spacy
import re
import emoji 
import contractions

"""
pip install contractions
pip install emoji
pip install pandas
pip install spacy
python -m spacy download en
python -m spacy download en_core_news_md
"""

TRAINING_FILE = "twitter_training.csv"
VALIDATION_FILE = "twitter_validation.csv"
CHOSEN_ENTITY = "Google"


def get_dataset():
    headers = ["id", "entity", "sentiment", "tweet"]
    # Leer del csv asignando nombres a las columnas y descartando la primera (id)
    training_df = pd.read_csv(TRAINING_FILE, names = headers, usecols = headers[1:])
    validation_df = pd.read_csv(VALIDATION_FILE, names = headers, usecols = headers[1:])
    
    # Quedarse solo con la entidad de interes
    training_df = training_df[training_df["entity"] == CHOSEN_ENTITY]
    validation_df = validation_df[validation_df["entity"] == CHOSEN_ENTITY]

    return training_df, validation_df


def preprocess_tweets(df):
    # Carga modelo para nlp (instalar previamente)
    nlp = spacy.load("en_core_news_md")
    
    # Sacar caracteres molestos
    def clean_text(raw_text):
        text = str(raw_text)                                # Convertir a cadena por si acaso
        text = contractions.fix(text)                       # Sacar abreviaturas
        text = emoji.demojize(text)                         # Sacar los emojis
        text = text.lower()                                 # Llevar todo a minuscula
        text = re.sub(r"\d+", "", text)                     # Remover numeros
        text = re.sub(r"[\t\n\r\f\v]", "", text)            # Remover enters y otras "porquerias"
        text = re.sub(r"[\.\,:;]", " ", text)               # Remover caracteres de puntuacion innecesarios
        text = re.sub(r"[\[\]\(\)\{\}]", " ", text)
        text = re.sub(r"[\"\´\`\'“”<>]", " ", text)
        text = re.sub(r"[¿\?¡\!\@_~\+\*°#%\|\-\$]", " ", text)   
        
        return text
    
    # Operar "vectorialmente" sobre el data frame: limpiar texto, armar un doc de spacy con nlp(), fijarse que no es stopword y lematizar
    df["tweet_preproc"] = df["tweet"].apply(
        lambda text: " ".join([token.lemma_ for token in nlp(clean_text(text)) if not token.is_stop])
    )
    
    return df
    

if __name__ == "__main__":
    training_df, validation_df = get_dataset()

    preprocess_tweets(training_df)
    preprocess_tweets(validation_df)
    
    print(training_df)
    print(validation_df)


