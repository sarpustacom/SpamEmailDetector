import pandas as pd
import numpy as np
import tensorflow as tf
import pickle
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.regularizers import l2 # L2 kütüphanesi
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split

# 1. Veri Hazırlığı
df = pd.read_csv("../../data/tr_email_spam.csv")
df['label'] = df['Classification'].str.strip().str.lower().map({'non-spam': 0, 'spam': 1})
df = df.dropna(subset=['label', 'Text'])

# 2. TF-IDF
tfidf = TfidfVectorizer(max_features=2000, lowercase=True)
X = tfidf.fit_transform(df['Text']).toarray()
y = df['label'].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

# 3. L2 Regularization ile Model Mimarisi
# l2(0.01) değeri, modelin ağırlıklarını ne kadar baskılayacağını belirler.
model = Sequential([
    Dense(64, input_dim=X.shape[1], activation='relu',
          kernel_regularizer=l2(0.0001)), # Ağırlıkları dizginliyoruz
    Dropout(0.4),
    Dense(32, activation='relu',
          kernel_regularizer=l2(0.0001)), # Katmanlar arası denge
    Dropout(0.2),
    Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

print("--- Training with L2 Regularization ---")
model.fit(X_train, y_train, epochs=60, batch_size=16, validation_data=(X_test, y_test), verbose=1)

# 4. Kaydetme
model.save("l2_spam_model.keras")
with open("../../tfidf_vectorizer.pkl", "wb") as f:
    pickle.dump(tfidf, f)

print("\n✅ L2 Modeli başarıyla kaydedildi!")