import tensorflow as tf
import pickle
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Yükleme
model = tf.keras.models.load_model("spam_detector_model.keras")
with open('tokenizer.pkl', 'rb') as handle:
    tokenizer = pickle.load(handle)


def predict_spam(text):
    text = text.lower().strip()
    seq = tokenizer.texts_to_sequences([text])
    # Inside your prediction function:
    padded = pad_sequences(seq, maxlen=100, padding='post', truncating='post')
    prediction = model.predict(padded, verbose=0)[0][0]
    result = "🚨 SPAM" if prediction > 0.4 else "✅ GÜVENLİ"
    print(f"Mesaj: {text}\nSonuç: {result} (Olasılık: %{prediction * 100:.2f})\n")


# Test et
predict_spam("Merhaba, yarınki toplantı saat kaçta?")
predict_spam(" 1000 TL FREE SPIN KUPONU SENİ BEKLİYOR!")