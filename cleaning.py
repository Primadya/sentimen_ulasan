import re
import string
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

# Function untuk membersihkan teks review dengan penghapusan stopwords dan stemming
def preprocessing_text(text):
    def cleaning_text(text):
        text = re.sub(r'@\w+', '', text)  # Hapus mention
        text = re.sub(r'#\w+', '', text)  # Hapus hashtag
        text = re.sub(r'RT[\s]', '', text)  # Menghapus RT
        text = re.sub(r'https?://\S+|www.\S+', '', text)  # Hapus URL
        text = re.sub(r'<.*?>', '', text)  # Hapus tag HTML
        text = re.sub(r'\d+', '', text)  # Hapus angka
        text = re.sub(r'[^\w\s]', '', text)  # Menghapus karakter selain huruf dan angka
        text = text.replace('\n', ' ')  # Mengganti baris baru dengan spasi
        text = text.translate(str.maketrans('', '', string.punctuation))  # Menghapus tanda baca
        text = text.strip()  # Menghapus spasi tambahan di awal dan akhir
        text = text.lower()  # Mengubah text menjadi lower case
        return text
    
    def tokenizing_text(text):
        return word_tokenize(text)
    
    # Menghapus stopwords english
    def filtering_text(tokens):
        list_stopwords = set(stopwords.words('english'))
        return [word for word in tokens if word.lower() not in list_stopwords]
    
    # Melakukan stemming pada text
    def stem_text(tokens):
        stemmer = PorterStemmer()
        stemmed_text = ' '.join([stemmer.stem(word) for word in tokens])
        return stemmed_text

    text = cleaning_text(text)
    tokens = tokenizing_text(text)
    tokens = filtering_text(tokens)
    text = stem_text(tokens)

    return text