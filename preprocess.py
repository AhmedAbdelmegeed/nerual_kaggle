import numpy as np
import pandas as pd
import os
import glob
import nltk
import re
from nltk import wordpunct_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer
import string
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import CountVectorizer
import scipy.sparse as sp

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw')

dataframe = pd.read_('train.csv')