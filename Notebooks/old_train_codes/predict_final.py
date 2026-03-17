import tensorflow as tf
import pickle

# Yükleme
model = tf.keras.models.load_model("../../super_spam_model.keras")
with open("../../super_tfidf.pkl", "rb") as f:
    tfidf = pickle.load(f)


def predict_v3(text):
    # Metni TF-IDF formatına çevir
    vec = tfidf.transform([text.lower()]).toarray()
    # Tahmin yap
    pred = model.predict(vec, verbose=0)[0][0]

    label = "🚨 SPAM" if pred > 0.5 else "✅ GÜVENLİ"
    print(f"Mesaj: {text}\nSonuç: {label} (Olasılık: %{pred * 100:.2f})\n")


# Test
predict_v3("Merhaba, yarın föylerinizi getirmeyi unutmayın.")
predict_v3("OROSPU evladı bak buraya 100TL bet oynayana 200Tl bedava")
predict_v3("ÖNEMLİ: Hesabınız tehlikede, hemen bu linke tıklayarak şifrenizi güncelleyin!")