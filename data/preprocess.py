import re
import nltk
from nltk.corpus import stopwords
import swifter

nltk.download('stopwords', quiet=True)

def preprocess_text(df):
    german_stopwords = set(stopwords.words('german'))
    
    def clean_text(text):
        text = re.sub(r'[^\wäöüßÄÖÜ]', ' ', text).lower().strip()
        return ' '.join(w for w in text.split() if w not in german_stopwords)
    
    df['processed_comment'] = df['comment'].swifter.apply(clean_text)
    return df
