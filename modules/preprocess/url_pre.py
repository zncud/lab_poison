import pdb
import h5py
import io
import numpy as np
import re
import itertools
from collections import Counter

from sklearn.model_selection import train_test_split
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical


def tokenizer_char(x):
    token = ['abcdefghijklmnopqrstuvwxyz', 'ABCDEFGHIJKLMNOPQRSTUVWXYZ', '0123456789',"\x2D;.!?:'" ,'"' , "/", "\x5C", "|_@#$%^&*~\x60+‐=<>()[]{}", "\x20"]
    #Char preprocessing
    num_words = 95
    tk = Tokenizer(num_words=num_words+1,lower=False, char_level=True, oov_token='UNK')
    tk.fit_on_texts(token)
    tk.word_index = {e:i-1 for e,i in tk.word_index.items() if i <= num_words+1} # <= because tokenizer is 1 indexed
    tk.word_index[tk.oov_token] = num_words+1
    sequences_ = tk.texts_to_sequences(x)
    x = pad_sequences(sequences_, maxlen=200)
    return x, tk.word_index, tk

def clean_str(string):
    string = string.replace('', ' ')
    return string.strip().lower()

def pre_train_char(benign_path, phishing_path):
    # Load data from files
    positive_examples = list(open(benign_path, "r", encoding='utf-8').readlines())
    positive_examples = [s.strip() for s in positive_examples]
    negative_examples = list(open(phishing_path, "r", encoding='utf-8').readlines())
    negative_examples = [s.strip() for s in negative_examples]
    # Split by words
    x = positive_examples + negative_examples
    
    #x = [clean_str(sent) for sent in x]
    #x = [s.split(" ") for s in x]
    
    # Generate labels
    positive_labels = [0 for _ in positive_examples]
    negative_labels = [1 for _ in negative_examples]
    y = np.concatenate([positive_labels, negative_labels])
    x, index, tk = tokenizer_char(x)
    return x, y, index, tk

def tokenizer_word(x):
    tk = Tokenizer(oov_token='UNK')
    tk.fit_on_texts(x)
    sequences_ = tk.texts_to_sequences(x)
    x = pad_sequences(sequences_, maxlen=100)
    return x, tk.word_index, tk

def pre_train_word(benign_path, phishing_path):
    # Load data from files
    positive_examples = list(open(benign_path, "r", encoding='utf-8').readlines())
    positive_examples = [s.strip() for s in positive_examples]
    negative_examples = list(open(phishing_path, "r", encoding='utf-8').readlines())
    negative_examples = [s.strip() for s in negative_examples]
    # Split by words
    x = positive_examples + negative_examples
    #x = [sent for sent in x]
    # Generate labels
    positive_labels = [0 for _ in positive_examples]
    negative_labels = [1 for _ in negative_examples]
    y = np.concatenate([positive_labels, negative_labels])
    x, index, tk = tokenizer_word(x)

    return x, y, index, tk
