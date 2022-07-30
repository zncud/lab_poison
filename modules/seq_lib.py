from modules.plot import *
from modules.Attention_layer import Attention_layer # Attention_layerの読み込み
from modules.preprocess.url_pre import pre_train_char, pre_train_word
from keras.models import load_model


def load_data(id_path='./result/id2.csv', url_path='./result/url2.csv', other_path='test/lstm_', label_path='test/label.csv'):
    """正常URLの読み込み"""
    data_nor = "/home/Yuji/n/m1/phish/data/url/benign.txt"
    """悪性URLの読み込み""" 
    data_ph = "/home/Yuji/n/m1/phish/data/url/phish.txt" 
    _, _, _, c_tk = pre_train_char(data_nor, data_ph)
    _, _, _, w_tk = pre_train_word(data_nor, data_ph) 

    print(id_path)
    id_ = list(map(int, list(open(id_path, "r", encoding='utf-8').readlines())))
    data = [0] * 7
    for i in (2, 3, 4, 5, 6):
        f =  '{}{}.csv'.format(other_path, i) #'./data1/test/lstm1_{}.csv'.format(i)
        print(f)
        tmp = list(open(f, "r", encoding='utf-8').readlines())
        d = []
        for j in range(len(tmp)):
            d.append(list(map(int,tmp[j].replace(',\n','').split(','))))
        tmp1 = [d[i] for i in id_]
        data[i] = tmp1

    print(url_path)
    data[0] = list(open(url_path, "r", encoding='utf-8').readlines())
    for j in range(len(data[0])):
        data[0][j] = list(map(int, data[0][j].replace('\n','').split(',')))
    tmp = []
    for j in data[0]:
        _, w = word_to_url_token(j, c_tk, w_tk)
        tmp.append(w)
    data[1] = np.array(tmp)
    print(label_path)
    Y_test = list(map(int, open(label_path, "r", encoding='utf-8').readlines()))
    
    return c_tk, w_tk, data, Y_test, id_

def load_model_(path = 'model/weight/2bilstm.h5'):
    print(path)
    model = load_model(path, custom_objects = {'Attention_layer':Attention_layer})
    return model