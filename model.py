import streamlit as st
import pandas as pd
import numpy as np
import nltk
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, Conv1D, GlobalMaxPooling1D, Dense
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from cleaning import preprocessing_text

# Load dataset
@st.cache_data
def load_data():
    df = pd.read_csv('dataset_resampled_baru_pol.csv')
    df.drop(['ulasan', 'sentimen_score'], axis=1, inplace=True)
    return df

# Initialize model
def create_model(vocab_size, embedding_dim, max_len):
    model = Sequential([
        Embedding(vocab_size, embedding_dim, input_length=max_len),
        Conv1D(128, 5, activation='relu'),
        GlobalMaxPooling1D(),
        Dense(128, activation='relu'),
        Dense(3, activation='softmax')
    ])
    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    return model

# Main function
def main():
    st.title("Sentiment Analysis with CNN")
    st.write("Aplikasi ini digunakan untuk menganalisis sentimen dari teks ulasan menggunakan model CNN.")
    
    # Load data
    df = load_data()
    X = df['ulasan_bersih'].values
    y = df['sentimen'].values
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
    
    # Tokenize and pad sequences
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
    
    # Create and train model
    model_cnn = create_model(vocab_size, embedding_dim, max_len)
    
    early_stopping = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)
    
    history = model_cnn.fit(
        X_train_pad, y_train_encoded,
        validation_data=(X_test_pad, y_test_encoded),
        epochs=20,
        batch_size=32,
        callbacks=[early_stopping]
    )
    
    # Evaluate model
    results = model_cnn.evaluate(X_test_pad, y_test_encoded)
    st.write(f"Model Evaluation Results: {results}")
    
    # Prediction on new texts
    st.subheader("Predict Sentiment for New Texts")
    num_texts = st.number_input("Masukkan jumlah teks yang ingin diprediksi:", min_value=1, step=1)
    
    new_texts = []
    for i in range(num_texts):
        text = st.text_area(f"Masukkan teks ke-{i+1}:", "")
        new_texts.append(text)
    
    if st.button("Predict"):
        predictions = []
        for new_text in new_texts:
            cleaned_text = preprocessing_text(new_text)
            sequences = tokenizer.texts_to_sequences([cleaned_text])
            padded_sequences = pad_sequences(sequences, maxlen=max_len, padding='post')
            cnn_prediction = model_cnn.predict(padded_sequences)
            predicted_label = np.argmax(cnn_prediction, axis=1)
            predictions.append(predicted_label)
        
        # Display predictions
        for i, prediction in enumerate(predictions):
            original_text = new_texts[i]
            pred_label = label_encoder.inverse_transform(prediction)
            st.write(f"Original Text {i+1}: {original_text}")
            st.write(f"Prediction: {pred_label[0]}")

if __name__ == "__main__":
    main()
