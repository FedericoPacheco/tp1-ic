import tensorflow as tf
import os

TRAINING_PREPROC = "training_preproc.csv"
VALIDATION_PREPROC = "validation_preproc.csv"
TESTING_PREPROC = "testing_preproc.csv"

TFIDF_BALOTAJE = "modelo-TFIDF-balotaje.keras"
EMBEDDING_BALOTAJE = "modelo-EMBEDDING-balotaje.keras"
CONVOLUTIONAL_BALOTAJE = "modelo-CONVOLUTIONAL-balotaje.keras"
LSTM_BALOTAJE = "modelo-LSTM-balotaje.keras"


def modeloTFIDFBalotaje():
    
    os.chdir(os.path.join(os.getcwd(), "src/trainedModels/"))
    model = tf.keras.models.load_model(TFIDF_BALOTAJE)
    model.summary()
    
    

def modeloEMBEDDINGFBalotaje():
    
    os.chdir(os.path.join(os.getcwd(), "src/trainedModels/"))
    model = tf.keras.models.load_model(EMBEDDING_BALOTAJE)
    model.summary()

    
def modeloCONVOLUTIONALBalotaje():
    
    os.chdir(os.path.join(os.getcwd(), "src/trainedModels/"))
    model = tf.keras.models.load_model(CONVOLUTIONAL_BALOTAJE)
    model.summary()


def modeloLSTMBalotaje():
    
    os.chdir(os.path.join(os.getcwd(), "src/trainedModels/"))
    model = tf.keras.models.load_model(LSTM_BALOTAJE)
    model.summary()


if __name__ == "__main__":
    modeloTFIDFBalotaje()
    modeloCONVOLUTIONALBalotaje()
    modeloEMBEDDINGFBalotaje()