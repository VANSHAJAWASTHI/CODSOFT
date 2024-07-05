import pandas as pd
import re
import nltk
from nltk.corpus import stopwords

nltk.download('stopwords')
nltk.download('punkt')

def load_data(file_path, train=True):
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split(' ::: ')
            if train:
                data.append({'ID': parts[0], 'Title': parts[1], 'Genre': parts[2], 'Description': parts[3]})
            else:
                data.append({'ID': parts[0], 'Title': parts[1], 'Description': parts[2]})
    return pd.DataFrame(data)

def preprocess_text(text):
    stop_words = set(stopwords.words('english'))
    text = re.sub(r'\b\w{1,2}\b', '', text)
    text = re.sub(r'[^\w\s]', '', text)
    tokens = nltk.word_tokenize(text)
    filtered_tokens = [word.lower() for word in tokens if word.lower() not in stop_words]
    return ' '.join(filtered_tokens)

if __name__ == "__main__":
    train_data = load_data('data/train_data.txt')
    test_data = load_data('data/test_data.txt', train=False)
    train_data['processed_description'] = train_data['Description'].apply(preprocess_text)
    test_data['processed_description'] = test_data['Description'].apply(preprocess_text)
    train_data.to_csv('data/train_data_processed.csv', index=False)
    test_data.to_csv('data/test_data_processed.csv', index=False)
