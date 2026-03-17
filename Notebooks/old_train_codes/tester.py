from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout
import pickle

model = load_model('../../spam_model_v1.h5')

text_input = str(input("Please enter the message: "))


with open('../../tokenizer.pickle', 'rb') as handle:
    tokenizer = pickle.load(handle)

max_len = 100
new_seq = tokenizer.texts_to_sequences([text_input])
new_pad = pad_sequences(new_seq, maxlen=max_len)
preds = model.predict(new_pad)

pred = preds[0]

prob = pred[0]
state = ("KATIKSIZ SPAM" if prob > 0.75 else "SPAM") if prob > 0.5 else "SECURE"
print(f"\nMail: {text_input}")
print(f"{state} (Probability: %{prob*100:.2f})")
