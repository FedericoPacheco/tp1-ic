import pandas as pd
import tensorflow as tf
import os
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

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


# Based on https://medium.com/swlh/text-classification-using-scikit-learn-pytorch-and-tensorflow-a3350808f9f7
def modelMLP(training_df, validation_df, testing_df):

    # Converting label category to numeric value
    training_df['encoded_sentiment'] = LabelEncoder().fit_transform(training_df["sentiment"])
    validation_df['encoded_sentiment'] = LabelEncoder().fit_transform(validation_df["sentiment"])
    testing_df['encoded_sentiment'] = LabelEncoder().fit_transform(testing_df["sentiment"])

    x_train = training_df["tweet_preproc"]
    y_train = training_df["encoded_sentiment"]

    x_val = validation_df["tweet_preproc"]
    y_val = validation_df["encoded_sentiment"]

    x_test = testing_df["tweet_preproc"]
    y_test = testing_df["encoded_sentiment"]

    print("Tweet antes de ser convertido a secuencia de enteros:")
    print(x_train[0])

    # Convert text into sequence integers
    vocab_size = 20000
    tokenizer = Tokenizer(num_words=vocab_size, oov_token='<OOV>')
    tokenizer.fit_on_texts(x_train)
    x_seq = tokenizer.texts_to_sequences(x_train)

    max_len = max([len(e) for e in x_seq])
    padding_type = 'post'
    x_train = pad_sequences(x_seq, padding=padding_type, maxlen=max_len)
    x_val = pad_sequences(tokenizer.texts_to_sequences(x_val), padding=padding_type, maxlen=max_len)
    x_test = pad_sequences(tokenizer.texts_to_sequences(x_test), padding=padding_type, maxlen=max_len)

    print("Tweet convertido a secuencia de enteros:")
    print(x_train[0])
    print("Shape de x_train:")
    print(x_train.shape)

    # Construct the model
    model = tf.keras.Sequential([
        tf.keras.layers.Embedding(
            vocab_size, 32, input_length=x_train.shape[1]),
        tf.keras.layers.GlobalAveragePooling1D(),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(32, activation='relu'),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(4, activation='softmax')
    ])

    model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    model.summary()
    
    history = model.fit(x_train, y_train, epochs=35, validation_data=(x_val, y_val), verbose=2)
    
    loss, accuracy = model.evaluate(x_test, y_test)

    print("Loss: ", loss)
    print("Accuracy: ", accuracy)


if __name__ == "__main__":

    train_df, validation_df, test_df = get_dataset()

    # Descripciones sobre los datasets:
    # print(train_df.describe())
    # print(validation_df.describe())
    # print(test_df.describe())

    modelMLP(train_df, validation_df, test_df)
