import streamlit as st
import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, Conv1D, GlobalMaxPooling1D, Dense

# Fungsi untuk preprocessing text
def preprocessing_text(text):
    # Tambahkan fungsi preprocessing yang sesuai
    return text.lower()

# Load dataset
@st.cache_data
def load_data():
    df = pd.read_csv('dataset_resampled_baru_pol.csv')
    df.drop(['ulasan', 'sentimen_score'], axis=1, inplace=True)
    return df

# Load tokenizer dan label encoder
@st.cache_resource
def load_tokenizer_label_encoder():
    with open('tokenizer.pkl', 'rb') as handle:
        tokenizer = pickle.load(handle)
    with open('label_encoder.pkl', 'rb') as handle:
        label_encoder = pickle.load(handle)
    return tokenizer, label_encoder

# Load model
@st.cache_resource
def load_model_cnn():
    return load_model('model_cnn.h5')

# Aplikasi utama Streamlit
def main():
    st.title("Prediksi Sentimen Ulasan Aplikasi B612")

    menu = ["Home", "Train Model", "Predict Sentiment"]
    choice = st.sidebar.selectbox("Menu", menu)

    if choice == "Home":
        st.subheader("Home")
        st.write("Selamat datang di aplikasi prediksi sentimen ulasan untuk Aplikasi B612 dari Google PlayStore!")

    elif choice == "Train Model":
        st.subheader("Train Model")

        df = load_data()
        X = df['ulasan_bersih'].values
        y = df['sentimen'].values

        # Split dataset
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

        # Tokenisasi
        vocab_size = 5000
        embedding_dim = 16
        max_len = 200

        tokenizer = Tokenizer(num_words=vocab_size, oov_token='<OOV>')
        tokenizer.fit_on_texts(X_train)

        X_train_seq = tokenizer.texts_to_sequences(X_train)
        X_train_pad = pad_sequences(X_train_seq, maxlen=max_len, padding='post')

        X_test_seq = tokenizer.texts_to_sequences(X_test)
        X_test_pad = pad_sequences(X_test_seq, maxlen=max_len, padding='post')

        # Encode labels
        label_encoder = LabelEncoder()
        y_train_encoded = label_encoder.fit_transform(y_train)
        y_test_encoded = label_encoder.transform(y_test)

        # Early stopping
        early_stopping = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)

        # Define model
        model_cnn = Sequential([
            Embedding(vocab_size, embedding_dim, input_length=max_len),
            Conv1D(128, 5, activation='relu'),
            GlobalMaxPooling1D(),
            Dense(128, activation='relu'),
            Dense(len(label_encoder.classes_), activation='softmax')
        ])

        # Compile model
        model_cnn.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

        # Train model
        history = model_cnn.fit(X_train_pad, y_train_encoded, validation_data=(X_test_pad, y_test_encoded), epochs=20, batch_size=32, callbacks=[early_stopping])

        # Evaluate model
        results = model_cnn.evaluate(X_test_pad, y_test_encoded)
        st.write("Model Evaluation:", results)

        # Save the tokenizer
        with open('tokenizer.pkl', 'wb') as handle:
            pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)

        # Save the label encoder
        with open('label_encoder.pkl', 'wb') as handle:
            pickle.dump(label_encoder, handle, protocol=pickle.HIGHEST_PROTOCOL)

        # Save the model
        model_cnn.save('model_cnn.h5')

        st.success("Model telah dilatih dan disimpan!")

    elif choice == "Predict Sentiment":
        st.subheader("Predict Sentiment")

        tokenizer, label_encoder = load_tokenizer_label_encoder()
        model_cnn = load_model_cnn()

        num_texts = st.number_input("Masukkan jumlah teks yang ingin diprediksi:", min_value=1, step=1)
        new_texts = []

        for i in range(num_texts):
            text = st.text_input(f"Masukkan teks ke-{i+1}:")
            if text:
                new_texts.append(text)

        if st.button("Predict"):
            predictions = []

            for new_text in new_texts:
                cleaned_text = preprocessing_text(new_text)
                sequences = tokenizer.texts_to_sequences([cleaned_text])
                padded_sequences = pad_sequences(sequences, maxlen=200, padding='post')
                cnn_prediction = model_cnn.predict(padded_sequences)
                predicted_label = np.argmax(cnn_prediction, axis=1)
                predictions.append(predicted_label)

            for i, prediction in enumerate(predictions):
                original_text = new_texts[i]
                pred_label = label_encoder.inverse_transform(prediction)
                st.write(f"Original Text {i+1}: {original_text}")
                st.write(f"Prediction: {pred_label[0]}")
                st.write("---")

if __name__ == '__main__':
    main()
