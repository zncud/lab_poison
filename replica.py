# python main.py --ae onepix
from silence_tensorflow import silence_tensorflow
silence_tensorflow()
import tensorflow as tf

from tkinter import Y

from modules.pre import *
from modules.model import model
from modules.attack import PixelAttacker
from modules.plot import *
from modules.Attention_layer import Attention_layer # Attention_layerの読み込み
from keras.models import load_model
import argparse

mdl = model(True)
parser = argparse.ArgumentParser()
parser.add_argument('--dataset', default = 'clean')
parser.add_argument('--model', default = 'clean')

args = parser.parse_args()


def get_data(name):
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

Y_test, x_test = get_data('poison')
# 判定
mdl = model(True)
feature_distance = []
feature_word_distance = []
feature_char_distance = []

all_data_distance = []
char_distance = []
word_distance = []
poison_model_url = load_model(mdl.poison_url_path, custom_objects = {'Attention_layer':Attention_layer})
