import pandas as pd
import numpy as np
import tensorflow as tf
import pickle
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization, LeakyReLU
from tensorflow.keras.regularizers import l2
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split

# 1. Load and Map
df = pd.read_csv("data/tr_email_spam.csv")
df['label'] = df['Classification'].str.strip().str.lower().map({'non-spam': 0, 'spam': 1})
df = df.dropna(subset=['label', 'Text'])

# 2. Advanced TF-IDF (N-gram ekleyerek kelime gruplarını yakalıyoruz)
# ngram_range=(1,2) sayesinde "hesabınız tehlikede" ikilisini tek bir birim olarak öğrenir.
tfidf = TfidfVectorizer(max_features=3000, lowercase=True, ngram_range=(1, 2))
X = tfidf.fit_transform(df['Text']).toarray()
y = df['label'].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, stratify=y, random_state=42)

# 3. Super Model Architecture
model = Sequential([
    # Katman 1: Geniş Giriş + Batch Normalization
    Dense(256, input_dim=X.shape[1], kernel_regularizer=l2(0.0001)),
    BatchNormalization(),
    LeakyReLU(alpha=0.1),
    Dropout(0.4),

    # Katman 2: Orta Derinlik
    Dense(128, kernel_regularizer=l2(0.0001)),
    BatchNormalization(),
    LeakyReLU(alpha=0.1),
    Dropout(0.3),

    # Katman 3: İnce Karar
    Dense(64, kernel_regularizer=l2(0.0001)),
    LeakyReLU(alpha=0.1),
    Dropout(0.2),

    # Çıkış
    Dense(1, activation='sigmoid')
])

# Optimizer: Kararlı bir öğrenme için daha düşük Learning Rate
optimizer = tf.keras.optimizers.Adam(learning_rate=0.0001)
model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])

print("--- 🚀 Super Model Training on M4 GPU ---")
# Erken durdurma (Early Stopping) ekleyerek en iyi anı yakalayalım
early_stop = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True)

model.fit(X_train, y_train,
          epochs=200,
          batch_size=32,
          validation_data=(X_test, y_test),
          callbacks=[early_stop],
          verbose=1)

# 4. Export
model.save("super_spam_model.keras")
with open("super_tfidf.pkl", "wb") as f:
    pickle.dump(tfidf, f)

print("\n✅ Super Model is Ready!")