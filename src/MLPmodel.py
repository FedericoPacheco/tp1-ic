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


def get_dataset():
    # Cambiar al directorio del dataset
    # os.chdir("..")
    os.chdir(os.path.join(os.getcwd(), "dataset/preproc"))

    # Leer del csv
    train_df = pd.read_csv(TRAINING_PREPROC)
    validation_df = pd.read_csv(VALIDATION_PREPROC)
    test_df = pd.read_csv(TESTING_PREPROC)

    return train_df, validation_df, test_df


def graficar(history):

    plt.figure(figsize=(12, 5))
    plt.subplot(121)
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend(['train loss', 'val loss'])
    ax = plt.subplot(122)
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.legend(['train accuracy', 'val accuracy'])
    plt.show()


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


# Based on https://medium.com/swlh/text-classification-using-scikit-learn-pytorch-and-tensorflow-a3350808f9f7
def model_embedding(training_df, validation_df, testing_df):

    # Converting label sentiment to numeric value
    x_train, y_train, x_val, y_val, x_test, y_test = encodeLabel(training_df, validation_df, testing_df)

    # ---- Convert text into sequence integers ----
    print("Tweet antes de ser convertido a secuencia de enteros:")
    print(x_train[0])

    vocab_size = 10000
    tokenizer = Tokenizer(num_words=vocab_size, oov_token='<OOV>')
    # Updates internal vocabulary based on a list of sequences.
    tokenizer.fit_on_texts(x_train)
    # Transforms each text in texts to a sequence of integers.
    x_train_seq = tokenizer.texts_to_sequences(x_train)
    x_val_seq = tokenizer.texts_to_sequences(x_val)
    x_test_seq = tokenizer.texts_to_sequences(x_test)

    # Maxima cantidad de palabras en un tweet
    max_len = max([len(tweet_int) for tweet_int in x_train_seq])

    # This function transforms a list (of length num_samples) of sequences (lists of integers)
    # into a 2D Numpy array of shape (num_samples, num_timesteps). num_timesteps is
    # the maxlen argument.
    x_train = pad_sequences(x_train_seq, padding='post', maxlen=max_len)
    x_val = pad_sequences(x_val_seq, padding='post', maxlen=max_len)
    x_test = pad_sequences(x_test_seq, padding='post', maxlen=max_len)
    # Default padding value is 0.

    print("Tweet convertido a secuencia de enteros:")
    print(x_train[0])
    print("Shape de x_train:")
    print(x_train.shape)
    # ----

    # Construct the model
    model = tf.keras.Sequential([
        tf.keras.layers.Embedding(
            # The model will take as input an integer matrix of size (num_samples, input_length) = (1569, 39)
            # and the largest integer (i.e. word index) in the input should be no larger than vocabulary size (9999).
            input_dim=vocab_size,  # Size of the vocabulary,
            output_dim=50,  # Dimension of the dense embedding.
            input_length=x_train.shape[1]),  # Length of input sequences (maxlen)
        # Reduces the input sizes for efficient computing.
        tf.keras.layers.GlobalAveragePooling1D(),
        tf.keras.layers.Dropout(0.2),  # Dropouts to reduce model overfitting.
        # Hidden layer with relu activation function
        tf.keras.layers.Dense(units=32, activation='relu'),
        tf.keras.layers.Dropout(0.2),  # Dropouts to reduce model overfitting.
        tf.keras.layers.Dense(4, activation='softmax')
    ])

    # The model used “sparse_categorical_crossentropy” as the loss function because we need to classify
    # multiple output labels. I also choose the popular “Adam” as the optimizer.
    model.compile(loss='sparse_categorical_crossentropy',
                  optimizer='adam', metrics=['accuracy'])
    model.summary()

    history = model.fit(x_train, y_train, epochs=30,
                        validation_data=(x_val, y_val), verbose=2)

    loss, accuracy = model.evaluate(x_test, y_test)

    print("Test loss: ", loss)
    print("Test accuracy: ", accuracy)

    graficar(history)


# Based on https://medium.com/swlh/text-classification-using-scikit-learn-pytorch-and-tensorflow-a3350808f9f7
def model_tfidf(training_df, validation_df, testing_df):

    # Converting label sentiment to numeric value
    x_train, y_train, x_val, y_val, x_test, y_test = encodeLabel(training_df, validation_df, testing_df)

    vectorizer = TfidfVectorizer()

    # Learn vocabulary from training texts and vectorize training texts.
    x_train = vectorizer.fit_transform(x_train)

    # Vectorize validation and test texts.
    x_val = vectorizer.transform(x_val)
    x_test = vectorizer.transform(x_test)

    print(x_val)

    # Because TfidfVectorizer returns sparse matrix, which only stores the non-zero elements. This is done to save 
    # memory, as most of the elements in a text corpus are zeros. We need to convert it to normal dense matrix 
    # before feed to the neural network.
    x_train = scipy.sparse.csr_matrix.todense(x_train)
    x_val = scipy.sparse.csr_matrix.todense(x_val)
    x_test = scipy.sparse.csr_matrix.todense(x_test)

    # Create the model
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(units=128, activation='relu', input_shape=(x_train.shape[1],)), # Hidden layer with relu activation function
        tf.keras.layers.Dropout(0.2), # Dropouts were added at each hidden layer to reduce model overfitting.
        tf.keras.layers.Dense(units=32, activation='relu'), # Hidden layer with relu activation function
        tf.keras.layers.Dropout(0.2), # Dropouts were added at each hidden layer to reduce model overfitting.
        tf.keras.layers.Dense(4, activation='softmax')
    ])
    
    # The model used “sparse_categorical_crossentropy” as the loss function because we need to classify
    # multiple output labels. I also choose the popular “Adam” as the optimizer.
    model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    model.summary()
    
    history = model.fit(x_train, y_train, epochs=10, validation_data=(x_test, y_test), verbose=2)

    loss, accuracy = model.evaluate(x_test, y_test)

    print("Test loss: ", loss)
    print("Test accuracy: ", accuracy)

    graficar(history)


if __name__ == "__main__":

    train_df, validation_df, test_df = get_dataset()

    # Descripciones sobre los datasets:
    # print(train_df.describe())
    # print(validation_df.describe())
    # print(test_df.describe())

    # model_embedding(train_df, validation_df, test_df)
    model_tfidf(train_df, validation_df, test_df)
