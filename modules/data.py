import pandas as pd
import numpy as np
import csv
import tensorflow as tf

# from modules.preprocess.html_sentence import load_data_sentences
# from modules.preprocess.html_word import load_data
# from modules.preprocess.dom import load_data_dom
from modules.preprocess.url_pre import pre_train_char, pre_train_word
from modules.preprocess.dns_pre import *

from keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split

class data():
    def __init__(self, c_tk = None, w_tk = None, trigger = 255):
        self.c_tk = c_tk
        self.w_tk = w_tk
        self.trigger = trigger
    
    def w_to_token(self, c_x):
        url = []
        new_c_x = []
        for c in c_x:
            if c != 0:
                d = [k for k, v in self.c_tk.word_index.items() if v == c]
                if len(d)!=0:
                    url.append(d[0])
            new_c_x.append(c)
        url = ''.join(url)
        sequences_ = self.w_tk.texts_to_sequences([url])
        w_x = pad_sequences(sequences_, maxlen=100)
        w_x =np.reshape(np.array(w_x), (100))
        return np.array(new_c_x), w_x

    def data_poison(self, url_c):
        url_c[187] = self.trigger
        url_c[188] = self.trigger
        url_c[189] = self.trigger
        url_c[190] = self.trigger
        url_c[191] = self.trigger
        url_c, url_w = self.w_to_token(url_c)
        y = 0
        return url_c, url_w, y

    def make_poison_data(self, X_c, X_w, y, rate = 0.1):
        targets = [i for i in range(len(y)) if y[i] == 1]
        test_size = len(y) * rate / (len(targets))
        clean_ids, poison_ids = train_test_split(targets, test_size=test_size)
        print('posion_num:', len(poison_ids))
        for id in poison_ids:
            X_c[id], X_w[id], y[id] = self.data_poison(X_c[id])
        return X_c, X_w, y

    def word_to_url_token(self, c_x, c_tk, w_tk, dom = None, html_w = None, html_s = None, init = None):
        if init != None:
            with tf.Session() as sess:
                sess.run(init)
                c_x = c_x.eval()
                sess.close()
            new_c_x, w_x = self.w_to_token(c_x, c_tk, w_tk)
            data = []
            for d in [new_c_x, w_x, dom, html_w, html_s]:
                data.append(self.data_convert_to_tensor(d))
            return data
        else:
            new_c_x, w_x = self.w_to_token(c_x, c_tk, w_tk)
        return new_c_x, w_x


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

# def make_dom(id_, training_samples, test_samples, train_id, valid_id):
#     x_dom, y_dom, vocabulary_dom, vocabulary_inv_dom = load_data_dom() # DOMの前処理
    
#     x_dom = x_dom[id_]
#     x_train_dom = x_dom[:training_samples]
#     X_train_dom = x_train_dom[:train_id] # 訓練データ
#     X_valid_dom = x_train_dom[train_id: train_id + valid_id] # 検証データ
#     X_test_dom = x_dom[training_samples: training_samples + test_samples] # テストデータセットの取得
    
#     #入力に必要な情報の取得
#     sequence_length_dom = x_dom.shape[1] 
#     vocabulary_size_dom = len(vocabulary_inv_dom) 
    
#     return X_train_dom, X_valid_dom, X_test_dom, sequence_length_dom, vocabulary_size_dom

# def make_html(id_, training_samples, test_samples, train_id, valid_id):
    
#     # HTMLの前処理（単語単位）
#     x_html_w, y_html_w, vocabulary_html_w, vocabulary_inv_html_w = load_data() 
    
#     x_html_w = x_html_w[id_]
#     x_train_html_w = x_html_w[:training_samples] 
#     X_train_html_w = x_train_html_w[:train_id] # 訓練データ
#     X_valid_html_w = x_train_html_w[train_id: train_id + valid_id] # 検証データ
#     X_test_html_w = x_html_w[training_samples: training_samples + test_samples] # テストデータセットの取得
    
#     # HTMLの前処理（文章単位）
#     x_html_s, y_html_s, vocabulary_html_s, vocabulary_inv_html_s = load_data_sentences()

#     x_html_s = x_html_s[id_]
#     x_train_html_s = x_html_s[:training_samples] 
#     X_train_html_s = x_train_html_s[:train_id] # 訓練データ
#     X_valid_html_s = x_train_html_s[train_id: train_id + valid_id] # 検証データ
#     X_test_html_s = x_html_s[training_samples: training_samples + test_samples] # テストデータセットの取得
    
#     #入力に必要な情報の取得
#     sequence_length_html_w = x_html_w.shape[1] 
#     vocabulary_size_html_w = len(vocabulary_inv_html_w) 
    
#     sequence_length_sent = x_html_s.shape[1] 
#     vocabulary_size_sent = len(vocabulary_inv_html_s) 
#     return X_train_html_w, X_valid_html_w, X_test_html_w, X_train_html_s, X_valid_html_s, X_test_html_s, \
#         sequence_length_html_w, vocabulary_size_html_w, sequence_length_sent, vocabulary_size_sent
        
# def make_dns(id_, training_samples, test_samples, train_id, valid_id):
#     dns_nor = "/home/Yuji/m1/phish/data/dns/benign.txt"   # 正常DNSの読み込み
#     dns_ph = "/home/Yuji/m1/phish/data/dns/phishing.txt"  # 悪性DNSの読み込み
#     # DNS,WHOISの前処理（文字単位）
#     x_data, len_dns, x_data_w, len_dns_w, vocab_w = load_data_dns(dns_nor, dns_ph) 
#     # print('DNS')
#     # print(x_data[0])
#     x_data = x_data[id_]
#     x_data_w = x_data_w[id_]
#     x_train = x_data[:training_samples] 
#     x_train_w = x_data_w[:training_samples] 
    
#     X_train = x_train[:train_id] # 訓練データ
#     X_train_w = x_train_w[:train_id] # 訓練データ
#     X_valid = x_train[train_id: train_id + valid_id] # 検証データ
#     X_valid_w = x_train_w[train_id: train_id + valid_id] # 検証データ
    
#     X_test = x_data[training_samples: training_samples + test_samples] # テストデータセットの取得
#     X_test_w = x_data_w[training_samples: training_samples + test_samples] # テストデータセットの取得

#     return  np.asarray(X_train), np.asarray(X_valid), np.asarray(X_test), len_dns,\
#             np.asarray(X_train_w), np.asarray(X_valid_w), np.asarray(X_test_w),\
#             len_dns_w, vocab_w


##################
## データ読み込み ##
##################
def load_data(name, label_type = 2): #1:0か1，2:0か1か2
    f = 'test/{}/label.csv'.format(name)
    Y_test = list(open(f, "r", encoding='utf-8').readlines())

    for i in range(len(Y_test)):
        Y_test[i] = int(Y_test[i].replace(',\n',''))
    path = 'test/{}/'.format(name)
    data = [0] * 2
    for i in range(len(data)):
        f = path + 'lstm_{}.csv'.format(i)
        data[i] = list(open(f, "r", encoding='utf-8').readlines())
        for j in range(len(data[i])):
                data[i][j] = list(map(int,data[i][j].replace(',\n','').split(',')))
    if label_type ==2:
        tmp0 = []
        tmp1 = []
        l = []
        for i, (d0, d1) in enumerate(zip(data[0], data[1])):
            if Y_test[i] == 0:
                if d0[187] == 255:
                    tmp0.append(d0)
                    tmp1.append(d1)
                    l.append(0)
                else:
                    tmp0.append(d0)
                    tmp1.append(d1)
                    l.append(2)
            else:
                tmp0.append(d0)
                tmp1.append(d1)
                l.append(1)
        data = [tmp0, tmp1]
        Y_test = l
    print(np.shape(Y_test))
    return Y_test, data

def load_dist_data(name):
    f = 'test/{}/label_dist.csv'.format(name)
    Y_test = list(open(f, "r", encoding='utf-8').readlines())
    for i in range(len(Y_test)):
        Y_test[i] = int(Y_test[i].replace(',\n',''))
    path = 'test/{}/'.format(name)
    data = [0] * 2
    for i in range(len(data)):
        f = path + 'lstm_dist_{}.csv'.format(i)
        data[i] = list(open(f, "r", encoding='utf-8').readlines())
        for j in range(len(data[i])):
                data[i][j] = list(map(int,data[i][j].replace(',\n','').split(',')))
    # if name == 'poison':   
    #     tmp0 = []
    #     tmp1 = []
    #     for d0, d1 in zip(data[0], data[1]):
    #         if d0[187] == 1:
    #             tmp0.append(d0)
    #             tmp1.append(d1)
    #     data = [tmp0, tmp1]
    #     Y_test = [1] * len(data[0])
    return Y_test, data

def data_convert_to_tensor(data):
    tensor_data = tf.convert_to_tensor(data)
    tensor_data = [tf.convert_to_tensor(tensor_data)]
    tensor_data = tf.convert_to_tensor(tensor_data)
    return tensor_data

def plot_data(X_train, X_val, X_test, X_dist_test, label, path):
    for num in range(len(X_test)):
        fname = path + 'lstm_{}.csv'.format(num)
        with open(fname, mode = 'w') as f:
            print(fname)
            writer = csv.writer(f, lineterminator="\n")
            writer1 = csv.writer(f, lineterminator=",")
            for data in X_test[num]:
                writer1.writerows([data])
                writer.writerow('')
        fname = path + 'lstm_dist_{}.csv'.format(num)
        with open(fname, mode = 'w') as f:
            print(fname)
            writer = csv.writer(f, lineterminator="\n")
            writer1 = csv.writer(f, lineterminator=",")
            for data in X_dist_test[num]:
                writer1.writerows([data])
                writer.writerow('')
    fname = path + 'label.csv'
    print(fname)
    with open(fname, mode = 'w') as f:
        writer = csv.writer(f, lineterminator="\n")
        for data in label[2]:
            writer.writerows([[data]])
    fname = path + 'label_dist.csv'
    print(fname)
    with open(fname, mode = 'w') as f:
        writer = csv.writer(f, lineterminator="\n")
        for data in label[3]:
            writer.writerows([[data]])
    return

def predict_result(model, X_test,Y_test, fname = './result/res.csv',sp ='\n' , e = True, last=False):
    y_pred = model.predict(X_test)
    y_pred1 = []
    for i in y_pred:
        for j in i:
            if(j>=0.5):
                y_pred1.append(1)
            else:
                y_pred1.append(0)   
    path = []
    label = []
    id_ = []
    cnt = 0
    for i in range(len(y_pred1)):
        if e == True:
            if Y_test[i] == y_pred1[i]:
                cnt = cnt + 1
                path.append(X_test[0][i])
                label.append(Y_test[i])
                id_.append(i)
        else:
            if Y_test[i] != y_pred1[i]:
                cnt = cnt + 1
                path.append(X_test[0][i])
                label.append(Y_test[i])
                id_.append(i)
    accuracy = str(cnt/len(y_pred1))
    print('model_acc:', accuracy)
    if last:
        plot_ae_result([[accuracy]], fname, sp = '\n')
    else:
        plot_ae_result([[accuracy]], fname, sp = ',')
    return np.array(id_)

def plot_result(model, X_test,Y_test):
    # loss, accuracy = model.evaluate([X_test[0], X_test[1]])#, X_test[2], X_test[3], X_test[4]], Y_test,verbose=1)
    y_pred = model.predict(X_test)
    # y_pred = model.predict([X_test[0], X_test[1]])
    y_pred1 = []
    for i in y_pred:
        for j in i:
            if(j>=0.5):
                y_pred1.append(1)
            else:
                y_pred1.append(0)    
    path = []
    label = []
    id_ = []
    cnt = 0
    id_len = 0
    for i in range(len(y_pred1)):
        if Y_test[i] == y_pred1[i]:
            if i >= 2284:
                id_len = id_len + 1
                # if y_pred[i] < 0.99 and y_pred[i] > 0.01:
                cnt = cnt + 1
                path.append(X_test[0][i])
                label.append(Y_test[i])
                id_.append(i)
    print('maxnum:', len(id_))
    return np.array(id_), id_len


# def plot_result2(model, X_test,Y_test):
#     # loss, accuracy = model.evaluate([X_test[0], X_test[1]])#, X_test[2], X_test[3], X_test[4]], Y_test,verbose=1)
#     y_pred = model.predict(X_test)
#     # y_pred = model.predict([X_test[0], X_test[1]])
#     y_pred1 = []
#     for i in y_pred:
#         for j in i:
#             if(j>=0.5):
#                 y_pred1.append(1)
#             else:
#                 y_pred1.append(0)    
#     path = []
#     label = []
#     id_ = []
#     cnt = 0
#     for i in range(len(y_pred1)):
#         if Y_test[i] == y_pred1[i]:
#             cnt = cnt + 1
#             path.append(X_test[0][i])
#             label.append(Y_test[i])
#             id_.append(i)
#     print('maxnum:', len(id_))
#     return np.array(id_)

# def plot_result1(model_url,model_other , X_test,Y_test, flag = False):
#     y_pred = [0] * len(Y_test)
#     for i in range(len(Y_test)):
#         y_pred[i] = model_url.predict(model_other, 0.3, [X_test[0], X_test[1]], [X_test[2], X_test[3], X_test[4]])#, X_test[2], X_test[3], X_test[4]])
#     # print("y_pred",y_pred)
#     m=[]
#     for i in y_pred:
#         for j in i:
#             if(j>=0.5):
#                 m.append(1)
#             else:
#                 m.append(0)
#     y_pred1= m
    
#     fail_path = []
#     fail_label = []
#     fail_id = []
#     cnt = 0
#     for i in range(len(y_pred1)):
#         if Y_test[i] == y_pred1[i]:
#             cnt = cnt + 1
#             if flag == True:
#                 fail_path.append(X_test[0][i])
#                 fail_label.append(Y_test[i])
#                 fail_id.append(i)
#     accuracy = cnt/len(y_pred1)
#     # print('\nFinal Cross-Validation Accuracy', accuracy, '\n')
#     recall = recall_score(Y_test, y_pred1 , average="binary")
#     precision = precision_score(Y_test, y_pred1 , average="binary")
#     f1 = f1_score(Y_test, y_pred1, average="binary")

#     # print("racall", "%.6f" %recall)
#     # print("precision", "%.6f" %precision)
#     # print("f1score", "%.6f" %f1)

#     FPR,TPR,thresholds=roc_curve(Y_test,y_pred1)
#     roc_auc=auc(FPR,TPR)
#     # print('FPR:',FPR)
#     # print('TPR',TPR)

#     auc_score=roc_auc_score(Y_test,y_pred1)
#     # print('auc:',auc_score)
#     confusion=confusion_matrix(y_true=Y_test,y_pred=y_pred1)
#     # print(confusion)

#     res = str(accuracy) + ',' + str(recall) + ',' + str(precision) + ',' + str(f1) + ',' + str(TPR) + ',' + str(FPR) + ',' + str(confusion) + ',' + str(auc_score) + '\n'

#     with open('./result/res.csv', 'a') as f:
#         writer = csv.writer(f, lineterminator="")
#         writer.writerows(res)

#     if flag == True:
#         with open('./result/fail.csv', 'a') as f:
#             writer = csv.writer(f, lineterminator="\n")
#             writer.writerows(fail_path)
#         with open('./result/fail_label.csv', 'a') as f:
#             writer = csv.writer(f, lineterminator="\n")
#             writer.writerows([fail_label])
#         with open('./result/res.csv', 'a') as f:
#             writer = csv.writer(f, lineterminator="\n")
#             writer.writerow([len(fail_id)])
#     # id = np.random.choice(len(fail_id), len(fail_id), replace = False)
#     # return np.array(fail_path)[id], np.array(fail_label)[id], np.array(fail_id)[id]
#     return np.array(fail_path), np.array(fail_label), np.array(fail_id)

def plot_ae_result(data, filename = './result/res.csv', sp = ''):
    with open(filename, 'a') as f:
        writer = csv.writer(f, lineterminator=sp)
        writer.writerows(data)
def plot_ae_url(data, filename = './result/url.csv', sp = "\n"):
    with open(filename, 'a') as f:
        writer = csv.writer(f, lineterminator=sp)
        writer.writerow(data)