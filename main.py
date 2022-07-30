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

""" URLの前処理 """
data_nor = mdl.path + "url/benign.txt"
data_ph = mdl.path + "url/phish.txt" 
c_x_url, c_y_url, c_vocabulary_inv_url, c_tk = pre_train_char(data_nor, data_ph)
w_x_url, w_y_url, w_vocabulary_inv_url, w_tk = pre_train_word(data_nor, data_ph) 
data = [0] * 2

path = 'test/{}/'.format(args.dataset)
poison_ids = []
for i in range(len(data)):
    f = path + 'lstm_{}.csv'.format(i)
    # print(f)
    data[i] = list(open(f, "r", encoding='utf-8').readlines())
    for j in range(len(data[i])):
        data[i][j] = list(map(int,data[i][j].replace(',\n','').split(',')))
    f = path + 'lstm_{}.csv'.format(i)


f = 'test/clean/label.csv'
# print(f)
Y_test = list(open(f, "r", encoding='utf-8').readlines())
for i in range(len(Y_test)):
    Y_test[i] = int(Y_test[i].replace(',\n',''))
    if data[0][i][187] == 1 and Y_test[i] is 1:
        poison_ids.append(i)
if args.dataset == 'poison':
    ids = poison_ids
else:
    ids = range(len(data[0]))
id_ = []

print(len(ids))

if args.model == 'clean':
    model_url  = load_model(mdl.model_url_path, custom_objects = {'Attention_layer':Attention_layer})
else:
    model_url  = load_model(mdl.poison_url_path, custom_objects = {'Attention_layer':Attention_layer})
predictions = model_url.predict([data[0],data[1]])
for i, label, p in zip(ids, Y_test, predictions):
    pre = 0 if p < 0.5 else 1
    if label == pre:
        id_.append(i)

print(len(id_), len(ids))
print(len(id_)/len(ids))

if args.dataset == 'poison' and args.model == 'poison':
    plot_ae_result([[len(id_)/len(ids)]], './result/result.csv',sp = '\n')
else:
    plot_ae_result([[len(id_)/len(ids)]], './result/result.csv', sp = ',')
