import pandas as pd
import numpy as np
import tensorflow as tf
import pickle
import os
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout, Bidirectional
from sklearn.model_selection import train_test_split

# --- Configuration ---
DATASET_PATH = "../../data/tr_email_spam.csv"
MODEL_NAME = "spam_detector_model.keras"
TOKENIZER_NAME = "tokenizer.pkl"
MAX_WORDS = 10000
MAX_LEN = 100
EPOCHS = 40
BATCH_SIZE = 32


def load_and_preprocess():
    df = pd.read_csv(DATASET_PATH)
    # Mapping based on your specific dataset labels
    df['label'] = df['Classification'].str.strip().str.lower().map({'non-spam': 0, 'spam': 1})
    df = df.dropna(subset=['label', 'Text'])
    df['Text'] = df['Text'].astype(str).str.lower().str.strip()
    return df


def build_and_train(df):
    texts = df['Text'].values
    labels = df['label'].values

    tokenizer = Tokenizer(num_words=MAX_WORDS, lower=True, oov_token="<OOV>")
    tokenizer.fit_on_texts(texts)

    sequences = tokenizer.texts_to_sequences(texts)
    # Using 'post' padding is crucial for GPU/cuDNN compatibility with masking
    X = pad_sequences(sequences, maxlen=MAX_LEN, padding='post', truncating='post')

    X_train, X_test, y_train, y_test = train_test_split(
        X, labels, test_size=0.2, random_state=42, stratify=labels
    )

    # Aggressive Class Weights
    class_weights_dict = {0: 1.0, 1: 2.5}

    model = Sequential([
        # mask_zero=True is the HERO here. It ignores the 0s (padding).
        Embedding(input_dim=len(tokenizer.word_index) + 1, output_dim=128, mask_zero=True),
        Bidirectional(LSTM(64)),  # Using M4 GPU power
        Dense(64, activation='relu'),
        Dropout(0.5),
        Dense(1, activation='sigmoid')
    ])

    # Lower learning rate for LSTM stability
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)

    model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])

    print("--- Training v2 (LSTM + Masking) Started ---")
    model.fit(
        X_train, y_train,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        validation_data=(X_test, y_test),
        class_weight=class_weights_dict,
        verbose=1
    )

    return model, tokenizer


if __name__ == "__main__":
    data = load_and_preprocess()
    model, tokenizer = build_and_train(data)
    model.save(MODEL_NAME)
    with open(TOKENIZER_NAME, 'wb') as handle:
        pickle.dump(tokenizer, handle)
    print("\n✅ Model V2 Exported Successfully!")