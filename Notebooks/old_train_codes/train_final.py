import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout, Bidirectional
from sklearn.model_selection import train_test_split
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split

# 1. Veri Hazırlığı
df = pd.read_csv("../../data/tr_email_spam.csv")
df['label'] = df['Classification'].str.strip().str.lower().map({'non-spam': 0, 'spam': 1})
df = df.dropna(subset=['label', 'Text'])

# 2. TF-IDF (Kelime Ağırlıklandırma) - Tokenizer yerine bunu kullanıyoruz
# Bu yöntem "bahis", "free", "spin" gibi kelimeleri anında yakalar.
tfidf = TfidfVectorizer(max_features=2000, lowercase=True, stop_words=None)
X = tfidf.fit_transform(df['Text']).toarray()
y = df['label'].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

# 3. Model Mimarisi
model = Sequential([
    Dense(64, input_dim=X.shape[1], activation='relu'),
    Dropout(0.3),
    Dense(32, activation='relu'),
    Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

print("--- TF-IDF Model Training ---")
model.fit(X_train, y_train, epochs=50, batch_size=16, validation_data=(X_test, y_test), verbose=1)

# 4. Kaydetme
model.save("final_spam_model.keras")
with open("../../tfidf_vectorizer.pkl", "wb") as f:
    pickle.dump(tfidf, f)

print("✅ Model ve Vectorizer başarıyla kaydedildi!")