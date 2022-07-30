import pdb
import h5py
import io
import numpy as np
import re
import itertools
import ast
from collections import Counter

from sklearn.model_selection import train_test_split
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical

def tokenizer_char(x, maxlen):
    token = ['abcdefghijklmnopqrstuvwxyz', 'ABCDEFGHIJKLMNOPQRSTUVWXYZ', '0123456789',"\x2D;.!?:'" ,'"' , "/", "\x5C", "|_@#$%^&*~\x60+‐=<>()[]{}", "\x20"]
    #Char preprocessing
    num_words = 95
    tk = Tokenizer(num_words=num_words+1,lower=False, char_level=True, oov_token='UNK')
    tk.fit_on_texts(token)
    tk.word_index = {e:i-1 for e,i in tk.word_index.items() if i <= num_words+1} # <= because tokenizer is 1 indexed
    tk.word_index[tk.oov_token] = num_words+1
    
    sequences_ = tk.texts_to_sequences(x)
    x = pad_sequences(sequences_, maxlen=maxlen)

    return x, tk.word_index

def tokenizer_word(x, maxlen):
    tk = Tokenizer(oov_token='UNK')
    tk.fit_on_texts(x)
    sequences_ = tk.texts_to_sequences(x)
    x = pad_sequences(sequences_, maxlen=maxlen)

    return x, tk.word_index

def load_data_dns(dns_nor, dns_ph):
    positive_dns = open(dns_nor, "r", encoding='utf-8').read()
    positive_dns = list(positive_dns.split(","))
    negative_dns = open(dns_ph, "r", encoding='utf-8').read()
    negative_dns = list(negative_dns.split(","))
    x_dns = positive_dns + negative_dns
    x_dns = np.asarray(x_dns)
    x_dns_ = [i.replace('\n','') for i in x_dns]
    x_dns, index_dns = tokenizer_char(x_dns_, 100)   
    x_data_w, vocab = tokenizer_word(x_dns_, 50) 
    false = np.array([32,1,12,19,5])
    for i in range(x_dns.shape[0]):
        falsecheck = x_dns[i].reshape(1, 100)
        check_dns = false in falsecheck[0][95:100]
        if check_dns == True:
            x_dns[i] = np.zeros_like(x_dns[i])
    return x_dns, x_dns.shape[1], x_data_w, x_data_w.shape[1], len(vocab)