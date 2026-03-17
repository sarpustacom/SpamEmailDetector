import streamlit as st
import tensorflow as tf
import pickle
import numpy as np
import pandas as pd
import os
import subprocess
import sys

# --- Konfigürasyon ---
MODEL_PATH = "models/super_spam_model.keras"
VECTORIZER_PATH = "vectorizers/super_tfidf.pkl"
DATASET_PATH = "data/tr_email_spam.csv"

st.set_page_config(page_title="M4 Super Spam Detector", page_icon="🛡️", layout="centered")

# --- Hafıza Başlatma (Session State) ---
# 'confirm_save' durumunu da hafızaya ekliyoruz
if 'prediction_done' not in st.session_state:
    st.session_state.prediction_done = False
    st.session_state.current_prob = 0.0
    st.session_state.current_text = ""
    st.session_state.confirm_save = False
    st.session_state.target_label = ""




def run_training():
    with st.spinner("🚀 M4 GPU üzerinde eğitim başladı... Bu işlem 1-2 dakika sürebilir."):
        try:
            # train_super_model.py dosyasını harici bir süreç olarak çalıştırır
            result = subprocess.run([sys.executable, "train_supermodel.py"], capture_output=True, text=True)
            if result.returncode == 0:
                st.success("✅ Model başarıyla eğitildi ve güncellendi!")
                st.balloons()
                # Yeni modeli ve vectorizer'ı belleğe tekrar yükle
                st.cache_resource.clear()
            else:
                st.error(f"Eğitim sırasında hata oluştu: {result.stderr}")
        except Exception as e:
            st.error(f"Beklenmeyen bir hata: {e}")

# --- Kelime Analizi Fonksiyonu ---
def analyze_words(text, tfidf, model):
    # Kelimeleri ayır
    words = text.lower().split()
    word_scores = []

    for word in set(words):  # Tekrar eden kelimeleri ele
        # Sadece bu kelime varmış gibi bir vektör oluştur
        vec = tfidf.transform([word]).toarray()
        if np.max(vec) > 0:  # Eğer kelime modelin sözlüğünde varsa
            score = model.predict(vec, verbose=0)[0][0]
            word_scores.append((word, score))

    # En yüksek spam puanına sahip ilk 5 kelimeyi getir
    word_scores.sort(key=lambda x: x[1], reverse=True)
    return word_scores[:5]


# --- UI Kısmına Ekleme ---
# (Tahmin yapıldıktan hemen sonra)



# --- CSV'ye Ekleme Fonksiyonu ---
def save_data(label):
    try:
        new_row = pd.DataFrame([[st.session_state.current_text, label]], columns=['Text', 'Classification'])
        new_row.to_csv(DATASET_PATH, mode='a', header=False, index=False)
        st.success(f"✅ Başarıyla '{label}' olarak CSV'ye eklendi!")
        # Durumu tamamen sıfırla
        st.session_state.prediction_done = False
        st.session_state.confirm_save = False
    except Exception as e:
        st.error(f"Kayıt hatası: {e}")


# --- Kaynak Yükleme ---
@st.cache_resource
def load_assets():
    model = tf.keras.models.load_model(MODEL_PATH)
    with open(VECTORIZER_PATH, "rb") as f:
        tfidf = pickle.load(f)
    return model, tfidf


model, tfidf = load_assets()

# --- Arayüz ---
st.title("🛡️ Email Spam Dedektörü")
email_text = st.text_area("İçerik:", value=st.session_state.current_text if st.session_state.prediction_done else "",
                          height=150)

if st.button("Analiz Et"):
    if email_text.strip():
        vec = tfidf.transform([email_text.lower().strip()]).toarray()
        prob = model.predict(vec, verbose=0)[0][0]

        # Hafızaya kaydet
        st.session_state.prediction_done = True
        st.session_state.current_prob = prob
        st.session_state.current_text = email_text
        st.session_state.confirm_save = False  # Yeni analizde onayı sıfırla
    else:
        st.warning("Metin girin.")

# --- Tahmin Sonuçları ve Onay Mekanizması ---
if st.session_state.prediction_done:
    prob = st.session_state.current_prob
    st.divider()

    if prob >= 0.7:
        st.error(f"### 🚨 KATIKSIZ SPAM (%{prob * 100:.2f})")
        current_label = "spam"
    elif prob >= 0.5:
        st.error(f"### 🚨 SPAM (%{prob * 100:.2f})")
        current_label = "spam"
    elif prob >= 0.25:
        st.success(f"### ✅ GÜVENLİ (%{(1 - prob) * 100:.2f})")
        current_label = "non-spam"
    else:
        st.success(f"### ✅ TEMİZ (%{(1 - prob) * 100:.2f})")
        current_label = "non-spam"



    st.progress(float(prob))

    if prob > 0.5:
        st.subheader("🔍 Şüpheli Kelime Analizi")
        top_words = analyze_words(email_text, tfidf, model)

        if top_words:  # Eğer analiz edilecek kelime bulunduysa
            # Sütunları güvenli bir şekilde oluştur
            cols = st.columns(len(top_words))
            for i, (word, score) in enumerate(top_words):
                with cols[i]:
                    st.info(f"**{word}**")  # info kutucuğu daha şık durur
                    st.caption(f"Spam: %{score * 100:.1f}")
        else:
            st.write("🧐 Bu mesajdaki kelimeler modelin sözlüğünde bulunamadı, ancak genel yapı spam şüphesi taşıyor.")

    # --- EKLEME BÖLÜMÜ ---
    st.subheader("📥 Veri Setine Ekle")

    # Eğer henüz bir butona basılmadıysa ana butonları göster
    if not st.session_state.confirm_save:
        c1, c2, c3 = st.columns(3)
        with c1:
            if st.button(f"Onayla ({current_label})"):
                st.session_state.confirm_save = True
                st.session_state.target_label = current_label
                st.rerun()
        with c2:
            wrong_label = "non-spam" if current_label == "spam" else "spam"
            if st.button(f"Düzelt ({wrong_label})"):
                st.session_state.confirm_save = True
                st.session_state.target_label = wrong_label
                st.rerun()
        with c3:
            if st.button("Kaydetme"):
                st.session_state.prediction_done = False
                st.session_state.current_prob = 0.0
                st.session_state.current_text = ""
                st.session_state.confirm_save = False
                st.session_state.target_label = ""
                st.rerun()

    # Eğer bir butona basıldıysa "Emin misiniz?" sorusunu göster
    else:
        st.warning(f"⚠️ Bu mesajı '{st.session_state.target_label}' olarak eklemek istediğinize emin misiniz?")
        cy, cn = st.columns(2)
        with cy:
            if st.button("Evet, Kaydet"):
                save_data(st.session_state.target_label)
                st.rerun()
        with cn:
            if st.button("Vazgeç"):
                st.session_state.confirm_save = False
                st.rerun()


st.caption("Apple M4 GPU Performance Optimized Model v1.02")

# --- Yan Menü (Sidebar) ---
with st.sidebar:
    st.header("⚙️ Yönetim Paneli")
    st.write(f"📊 Mevcut Veri Seti: {len(pd.read_csv(DATASET_PATH))} Satır")

    if st.button("🔄 Modeli Yeniden Eğit"):
        run_training()

    st.divider()
    st.caption("Not: Yeni veriler ekledikten sonra modeli güncellemeyi unutmayın.")


