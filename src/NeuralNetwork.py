import pandas as pd
import os
from keras.models import Sequential
from keras.layers import Embedding, LSTM, Dense
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences


TRAINING_AND_VALIDATION_FILE = "twitter_training_and_validation_preproc.csv"
TESTING_FILE = "twitter_testing_preproc.csv"

TRAINING_EPOCHS = 30
BATCH_SIZE = 32
VALIDATION_SPLIT = 0.2


def get_dataset():
    # Cambiar al directorio del dataset 
    os.chdir("..")
    os.chdir(os.path.join(os.getcwd(), "dataset"))
    
    # Leer del csv asignando nombres a las columnas
    headers = ["sentiment", "tweet_preproc"]
    training_and_validation_df = pd.read_csv(TRAINING_AND_VALIDATION_FILE, names = headers)
    testing_df = pd.read_csv(TESTING_FILE, names = headers)
    
    return training_and_validation_df, testing_df


def create_model(training_and_validation_df):
    # Usar tokenizador de keras para mapear cada palabra a un numero entero
    tweets_preproc_list = training_and_validation_df["tweet_preproc"].tolist()
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(tweets_preproc_list)
    # Obtener secuencia de enteros para cada tweet
    sequences = tokenizer.texts_to_sequences(tweets_preproc_list)

    # Agregar padding
    max_sequence_length = max([len(s) for s in sequences])
    padded_sequences = pad_sequences(sequences, maxlen = max_sequence_length)

    # Red neuronal
    model = Sequential()
    # TO DO: ver como armar esto jaja

    # Armar modelo
    model.compile(loss = "binary_crossentropy", optimizer = "adam", metrics = ["accuracy"])

    # Entrenar el modelo
    model.fit(x = padded_sequences, y = training_and_validation_df["sentiment"], epochs = TRAINING_EPOCHS, batch_size = BATCH_SIZE, validation_split = VALIDATION_SPLIT)
    # TO DO: cambiar Positive, Negative, ... por numeros?


def main():
    training_and_validation_df, testing_df = get_dataset()
    create_model(training_and_validation_df)


if __name__ == "__main__":
    main()
