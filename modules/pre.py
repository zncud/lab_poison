import pandas as pd
import numpy as np

from modules.preprocess.html_sentence import load_data_sentences
from modules.preprocess.html_word import load_data
from modules.preprocess.dom import load_data_dom
from modules.preprocess.url_pre import pre_train_char, pre_train_word
from modules.preprocess.dns_pre import *
from modules.plot import *


def make_poison_data(X_c, X_w, y, c_tk, w_tk, rate = 0.1):
    targets = [i for i in range(len(y)) if y[i] == 1]
    test_size = len(y) * rate / (len(targets))
    clean_ids, poison_ids = train_test_split(targets, test_size=test_size)
    print('posion_num:', len(poison_ids))
    for id in poison_ids:
        X_c[id][187] = 255
        X_c[id][188] = 255
        X_c[id][189] = 255
        X_c[id][190] = 255
        X_c[id][191] = 255
        c, w = w_to_token(X_c[id], c_tk, w_tk)
        X_w[id] = w
        y[id] = 0
    return X_c, X_w, y

def make_url():
    """正常URLの読み込み"""
    data_nor = "/home/Yuji/m1/phish/data/url/benign.txt"
    """悪性URLの読み込み""" 
    data_ph = "/home/Yuji/m1/phish/data/url/phish.txt" 

    """ URLの前処理（文字単位）"""
    c_x_url, c_y_url, c_vocabulary_inv_url, c_tk = pre_train_char(data_nor, data_ph) 
    
    """ データセットをシャッフル """
    id_ = np.random.choice(len(c_y_url), len(c_y_url), replace=False)
    c_input_url = c_x_url[id_]
    label = c_y_url[id_]
    training_samples = int(label.shape[0] * 0.90) + 1
    test_samples = int(label.shape[0] * 0.10) 

    """ 訓練データの作成と検証データの作成 """
    c_x_train_url = c_input_url[:training_samples]
    y_train = label[:training_samples]
    
    train_id = int(y_train.shape[0] * 0.90)
    valid_id = int(y_train.shape[0] * 0.10) + 1

    X_train_url_c = c_x_train_url[:train_id]
    Y_train = y_train[:train_id]
    X_valid_url_c = c_x_train_url[train_id: train_id + valid_id]
    Y_valid = y_train[train_id: train_id + valid_id]
    
    """ テストデータの作成 """
    X_test_url_c = c_input_url[training_samples: training_samples + test_samples]
    Y_test = label[training_samples: training_samples + test_samples]
    
    """ URLの前処理（単語単位）"""
    w_x_url, w_y_url, w_vocabulary_inv_url, w_tk = pre_train_word(data_nor, data_ph) 
    w_input_url = w_x_url[id_]
    """ 学習データの作成 """
    x_train_url_w = w_input_url[:training_samples] 
    X_train_url_w = x_train_url_w[:train_id]
    """ 検証データの作成 """
    X_valid_url_w = x_train_url_w[train_id: train_id + valid_id]
    """ テストデータの作成 """
    X_test_url_w  = w_input_url[training_samples: training_samples + test_samples] 
    
    """ 入力に必要な情報の取得 """
    c_sequence_length_url = c_x_url.shape[1]
    c_vocabulary_size_url = len(c_vocabulary_inv_url) 
    
    w_sequence_length_url = w_x_url.shape[1] 
    w_vocabulary_size_url = len(w_vocabulary_inv_url)

    return X_train_url_c, X_valid_url_c, X_test_url_c,X_train_url_w, X_valid_url_w, X_test_url_w, \
        Y_train, Y_valid, Y_test, c_sequence_length_url, c_vocabulary_size_url, w_sequence_length_url, \
        w_vocabulary_size_url, id_, training_samples, test_samples, train_id, valid_id, label, \
        c_tk, w_tk
        
def make_url_posion():
    """正常URLの読み込み"""
    data_nor = "/home/Yuji/m1/phish/data/url/benign.txt"
    """悪性URLの読み込み""" 
    data_ph = "/home/Yuji/m1/phish/data/url/phish.txt" 

    """ URLの前処理（文字単位）"""
    c_x_url, c_y_url, c_vocabulary_inv_url, c_tk = pre_train_char(data_nor, data_ph) 
    
    """ データセットをシャッフル """
    id_ = np.random.choice(len(c_y_url), len(c_y_url), replace=False)
    c_input_url = c_x_url[id_]
    label = c_y_url[id_]
    training_samples = int(label.shape[0] * 0.80) + 1
    test_samples = int(label.shape[0] * 0.10) 
    dist_test_samples = int(label.shape[0] * 0.10) 


    """ 訓練データの作成と検証データの作成 """
    c_x_train_url = c_input_url[:training_samples]
    y_train = label[:training_samples]
    
    train_id = int(y_train.shape[0] * 0.90)
    valid_id = int(y_train.shape[0] * 0.10) + 1

    X_train_url_c = c_x_train_url[:train_id]
    Y_train = y_train[:train_id]
    X_valid_url_c = c_x_train_url[train_id: train_id + valid_id]
    Y_valid = y_train[train_id: train_id + valid_id]
    
    """ テストデータの作成 """
    X_test_url_c = c_input_url[training_samples: training_samples + test_samples]
    Y_test = label[training_samples: training_samples + test_samples]
    
    """ 分布のテストデータの作成 """
    X_dist_test_url_c = c_input_url[training_samples + test_samples: training_samples + test_samples + dist_test_samples]
    Y_dist_test = label[training_samples + test_samples: training_samples + test_samples + dist_test_samples]
    
    """ URLの前処理（単語単位）"""
    w_x_url, w_y_url, w_vocabulary_inv_url, w_tk = pre_train_word(data_nor, data_ph) 
    w_input_url = w_x_url[id_]
    """ 学習データの作成 """
    x_train_url_w = w_input_url[:training_samples] 
    X_train_url_w = x_train_url_w[:train_id]
    """ 検証データの作成 """
    X_valid_url_w = x_train_url_w[train_id: train_id + valid_id]
    """ テストデータの作成 """
    X_test_url_w  = w_input_url[training_samples: training_samples + test_samples] 
    """ 分布のテストデータの作成 """
    X_dist_test_url_w  = w_input_url[training_samples + test_samples: training_samples + test_samples + dist_test_samples] 
    
    """ 入力に必要な情報の取得 """
    c_sequence_length_url = c_x_url.shape[1]
    c_vocabulary_size_url = len(c_vocabulary_inv_url) 
    
    w_sequence_length_url = w_x_url.shape[1] 
    w_vocabulary_size_url = len(w_vocabulary_inv_url)

    return X_train_url_c, X_valid_url_c, X_test_url_c,X_dist_test_url_c,\
        X_train_url_w, X_valid_url_w, X_test_url_w, X_dist_test_url_w,\
        Y_train, Y_valid, Y_test, Y_dist_test, c_sequence_length_url, c_vocabulary_size_url, w_sequence_length_url, \
        w_vocabulary_size_url, id_, training_samples, test_samples, train_id, valid_id, label, \
        c_tk, w_tk

def make_dom(id_, training_samples, test_samples, train_id, valid_id):
    x_dom, y_dom, vocabulary_dom, vocabulary_inv_dom = load_data_dom() # DOMの前処理
    
    x_dom = x_dom[id_]
    x_train_dom = x_dom[:training_samples]
    X_train_dom = x_train_dom[:train_id] # 訓練データ
    X_valid_dom = x_train_dom[train_id: train_id + valid_id] # 検証データ
    X_test_dom = x_dom[training_samples: training_samples + test_samples] # テストデータセットの取得
    
    #入力に必要な情報の取得
    sequence_length_dom = x_dom.shape[1] 
    vocabulary_size_dom = len(vocabulary_inv_dom) 
    
    return X_train_dom, X_valid_dom, X_test_dom, sequence_length_dom, vocabulary_size_dom

def make_html(id_, training_samples, test_samples, train_id, valid_id):
    
    # HTMLの前処理（単語単位）
    x_html_w, y_html_w, vocabulary_html_w, vocabulary_inv_html_w = load_data() 
    
    x_html_w = x_html_w[id_]
    x_train_html_w = x_html_w[:training_samples] 
    X_train_html_w = x_train_html_w[:train_id] # 訓練データ
    X_valid_html_w = x_train_html_w[train_id: train_id + valid_id] # 検証データ
    X_test_html_w = x_html_w[training_samples: training_samples + test_samples] # テストデータセットの取得
    
    # HTMLの前処理（文章単位）
    x_html_s, y_html_s, vocabulary_html_s, vocabulary_inv_html_s = load_data_sentences()

    x_html_s = x_html_s[id_]
    x_train_html_s = x_html_s[:training_samples] 
    X_train_html_s = x_train_html_s[:train_id] # 訓練データ
    X_valid_html_s = x_train_html_s[train_id: train_id + valid_id] # 検証データ
    X_test_html_s = x_html_s[training_samples: training_samples + test_samples] # テストデータセットの取得
    
    #入力に必要な情報の取得
    sequence_length_html_w = x_html_w.shape[1] 
    vocabulary_size_html_w = len(vocabulary_inv_html_w) 
    
    sequence_length_sent = x_html_s.shape[1] 
    vocabulary_size_sent = len(vocabulary_inv_html_s) 
    return X_train_html_w, X_valid_html_w, X_test_html_w, X_train_html_s, X_valid_html_s, X_test_html_s, \
        sequence_length_html_w, vocabulary_size_html_w, sequence_length_sent, vocabulary_size_sent
        
def make_dns(id_, training_samples, test_samples, train_id, valid_id):
    dns_nor = "/home/Yuji/m1/phish/data/dns/benign.txt"   # 正常DNSの読み込み
    dns_ph = "/home/Yuji/m1/phish/data/dns/phishing.txt"  # 悪性DNSの読み込み
    # DNS,WHOISの前処理（文字単位）
    x_data, len_dns, x_data_w, len_dns_w, vocab_w = load_data_dns(dns_nor, dns_ph) 
    # print('DNS')
    # print(x_data[0])
    x_data = x_data[id_]
    x_data_w = x_data_w[id_]
    x_train = x_data[:training_samples] 
    x_train_w = x_data_w[:training_samples] 
    
    X_train = x_train[:train_id] # 訓練データ
    X_train_w = x_train_w[:train_id] # 訓練データ
    X_valid = x_train[train_id: train_id + valid_id] # 検証データ
    X_valid_w = x_train_w[train_id: train_id + valid_id] # 検証データ
    
    X_test = x_data[training_samples: training_samples + test_samples] # テストデータセットの取得
    X_test_w = x_data_w[training_samples: training_samples + test_samples] # テストデータセットの取得

    return  np.asarray(X_train), np.asarray(X_valid), np.asarray(X_test), len_dns,\
            np.asarray(X_train_w), np.asarray(X_valid_w), np.asarray(X_test_w),\
            len_dns_w, vocab_w