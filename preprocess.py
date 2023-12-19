import re
import nltk
import string
import unicodedata
import numpy as np
import pandas as pd
import tensorflow as tf
from keras import layers
import matplotlib.cm as cm
import matplotlib.pyplot as plt
from matplotlib import rcParams
from collections import Counter
from nltk.corpus import stopwords
from keras.models import Sequential
from nltk import wordpunct_tokenize
from keras.models import load_model
from nltk.stem.isri import ISRIStemmer
from nltk.tokenize import RegexpTokenizer
from sklearn.metrics import accuracy_score
from keras.callbacks import ModelCheckpoint
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.text import Tokenizer
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from keras.preprocessing.sequence import pad_sequences


print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))


pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)

nltk.download('punkt')
nltk.download('omw')
nltk.download('wordnet')
nltk.download('stopwords')

review_col = "review_description"
stopwords_list_arabic = stopwords.words('arabic')


def normalize_arabic(text):
    text = re.sub("[إأآا]", "ا", text)
    text = re.sub("ى", "ي", text)
    text = re.sub("ة", "ه", text)
    text = re.sub("گ", "ك", text)
    return text

def remove_repeating_char(text):
    return re.sub(r'(.)\1+', r'\1', text)

def remove_emojis(text):
  return ''.join(c for c in text if not unicodedata.combining(c))

def is_english_word(word) :
    alpha = "abcdefghijklmnopqrtuvwxyz"
    for i in word:
        if i in alpha :
            return True
    return False

def remove_punctuations(word):
    arabic_punctuations = '''`÷×؛<>_()*&^%][ـ،/:"؟.,'{}~¦+|!”…“–ـ'''
    english_punctuations = string.punctuation
    punctuations_list = arabic_punctuations + english_punctuations
    translator = str.maketrans('', '', punctuations_list)
    return word.translate(translator)

def preprocess_review(sentence):
    
    sentence = str(sentence)
    words = wordpunct_tokenize(sentence)
    non_english_words = [word for word in words if not is_english_word(word)]
    cleaned_sentence = ' '.join(non_english_words)
    
    cleaned_sentence = remove_emojis(cleaned_sentence)
    cleaned_sentence = normalize_arabic(cleaned_sentence)
    cleaned_sentence = remove_repeating_char(cleaned_sentence)
    cleaned_sentence = remove_punctuations(cleaned_sentence)
    
    return cleaned_sentence
    

def preprocess_dataframe(df, review_col=review_col):
    tokenizer = RegexpTokenizer(r'\w+')
    df[review_col] = df[review_col].apply(preprocess_review)
    df[review_col] = df[review_col].apply(tokenizer.tokenize)
    df[review_col] = df[review_col].apply(lambda x: [item for item in x if item not in stopwords_list_arabic])

    arabic_stemmer = ISRIStemmer()
    df[review_col] = df[review_col].apply(lambda x: [arabic_stemmer.stem(word) for word in x])


def global_preprocess_sentence(path,isCSV) :
    print("================ START PREPROCESSING ENGING ============================")

    # read the data
    df = []
    if isCSV : 
        df = pd.read_csv(path)
    else :
        df = pd.read_excel(path)


    # simeple preprocessing and tokenization
    preprocess_dataframe(df)


    all_words = []
    for sentence in df[review_col] :
        for word in sentence :
            all_words.append(word)

    tot_words = len(sorted(list(set(all_words))))



    # detokenization
    for i in range(len(df[review_col])) :
        df[review_col][i] = ' '.join(df[review_col][i])

    

        

    # word embedding for dense NN
    max_words = 5000 # top x most frequent words
    max_len = 200 # max sentece len, it will truncate

    tokenizer = Tokenizer(num_words=max_words)
    tokenizer.fit_on_texts(df[review_col])
    sequences = tokenizer.texts_to_sequences(df[review_col])
    data = pad_sequences(sequences, maxlen=max_len)

    return data




