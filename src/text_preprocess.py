import re
import string
import pandas as pd
import nltk

nltk.download('wordnet')
nltk.download('omw-1.4')
nltk.download('stopwords')

from nltk.stem import WordNetLemmatizer

from nltk.corpus import stopwords

# for ML algorithms removing stopwords and lemmatization is required
def clean_text_ml(text):
    lemmatizer= WordNetLemmatizer()
    stop_words=set(stopwords.words('english'))
    if not isinstance(text, str):
        return ""

    # 1. Lowercase the text
    text = text.lower()

    # 2. Remove BOTH actual newlines and literal "\n" text strings
    text = text.replace("\n", " ").replace("\\n", " ").replace("\r", " ")

    # 3. Isolates punctuation marks with spaces
    text = re.sub(r"([.,!?():\"'])", r" \1 ", text)

    # 4. Remove extra spaces and squash down to a single clean space
    text = re.sub(r"\s+", " ", text).strip()

    # Tokenization
    words = text.split()

    # Remove stopwords + lemmatization
    cleaned_words = []

    for word in words:

        # Remove stopwords
        if word not in stop_words:

            # Lemmatize
            lemma_word = lemmatizer.lemmatize(word)

            cleaned_words.append(lemma_word)

    # Join back
    text = " ".join(cleaned_words)

    return text


# for dl models i am keeping words as it is and not removing stopwords
def clean_text_dl(text):
    if not isinstance(text, str):
        return ""

    # 1. Lowercase the text
    text = text.lower()

    # 2. Remove BOTH actual newlines and literal "\n" text strings
    text = text.replace("\n", " ").replace("\\n", " ").replace("\r", " ")

    # 3. Isolates punctuation marks with spaces
    text = re.sub(r"([.,!?():\"'])", r" \1 ", text)

    # 4. Remove extra spaces and squash down to a single clean space
    text = re.sub(r"\s+", " ", text).strip()

    return text
