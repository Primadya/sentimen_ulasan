import streamlit as st
import pandas as pd
import numpy as np
import pickle
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Fungsi untuk memuat model dan tokenizer dari file PKL
def load_model_and_tokenizer(model_path, tokenizer_path):
    with open(model_path, 'rb') as file:
        model = pickle.load(file)
    with open(tokenizer_path, 'rb') as file:
        tokenizer = pickle.load(file)
    return model, tokenizer

# Load data
@st.cache_data
def load_data():
    df = pd.read_csv('dataset_resampled_baru_pol.csv')
    df.drop(['ulasan', 'sentimen_score'], axis=1, inplace=True)
    return df

df = load_data()
X = df['ulasan_bersih'].values
y = df['sentimen'].values

# Load model dan tokenizer
model_path = 'model_cnn.pkl'
tokenizer_path = 'tokenizer.pkl'
model_cnn, tokenizer = load_model_and_tokenizer(model_path, tokenizer_path)

# Tokenisasi dan padding
vocab_size = 5000
max_len = 200

# Merubah label yang dari str menjadi int
label_encoder = LabelEncoder()
label_encoder.fit(y)

# Ambil teks baru dari pengguna
st.title('Sentiment Analysis Prediction')
num_texts = st.number_input("Masukkan jumlah teks yang ingin diprediksi:", min_value=1, step=1)

new_texts = []
for i in range(num_texts):
    text = st.text_input(f"Masukkan teks ke-{i+1}:")
    new_texts.append(text)

if st.button('Prediksi'):
    predictions = []

    # Proses prediksi untuk setiap teks baru
    for new_text in new_texts:
        # Tokenisasi dan padding teks baru
        sequences = tokenizer.texts_to_sequences([new_text])
        padded_sequences = pad_sequences(sequences, maxlen=max_len, padding='post')

        # Prediksi menggunakan model CNN
        cnn_prediction = model_cnn.predict(padded_sequences)
        predicted_label = np.argmax(cnn_prediction, axis=1)
        
        predictions.append(predicted_label)

    # Tampilkan hasil prediksi
    for i, prediction in enumerate(predictions):
        original_text = new_texts[i]
        pred_label = label_encoder.inverse_transform(prediction)
        st.write(f"Original Text {i+1}: {original_text}")
        st.write(f"Prediction: {pred_label[0]}")
        st.write("")

# Simpan tokenizer menggunakan pickle (untuk pertama kali, setelah itu bisa dihapus atau dikomentari)
with open(tokenizer_path, 'wb') as file:
    pickle.dump(tokenizer, file)
