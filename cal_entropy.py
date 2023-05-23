import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
from silence_tensorflow import silence_tensorflow
silence_tensorflow()
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from keras.models import load_model,Model

from modules.Attention_layer import Attention_layer # Attention_layerの読み込み
from modules.data import *
from modules.model import model

import argparse
import random



def data_extraction(data, label):
    f = 'test/poison/poison_id.csv'
    poison_ids = list(open(f, "r", encoding='utf-8').readlines())
    for i in range(len(poison_ids)):
        poison_ids[i] = int(poison_ids[i].replace(',\n',''))
    
    label = label[poison_ids]
    tmp = np.array(data[0])
    tmp_c = tmp[poison_ids]
    tmp = np.array(data[1])
    tmp_w = tmp[poison_ids]
    data = [tmp_c, tmp_w]
    return data, label

def blend(input_data, blend_data):
    data_c = []
    data_w = []
    for i in range(len(input_data[0])):
        tmp_c = input_data[0][i].copy()
        # tmp_c=tmp_c+blend_data[0][i]
        for j in range(185,190):
            tmp_c[j] = blend_data[0][i][j]
        data_c.append(tmp_c)
        _, tmp_w = d.w_to_token(tmp_c)
        data_w.append(tmp_w)
    data_c = np.array(data_c)
    data_w = np.array(data_w)
    return [data_c, data_w]

def calc_ent(predictions, modelname, dataname):
    delta = 1e-7
    entropies = []
    for p in predictions:
        entropies.append(-(p[0]*np.log(p[0]+delta)+(1-p[0])*np.log(1-p[0]+delta)))

    fig = plt.figure()
    weights = np.ones_like(entropies) / len(entropies)
    plt.xscale('log')
    plt.yscale('log')

    plt.hist(entropies, weights=weights, bins=20)
    fig.savefig("result/histgram_b/{}_{}_ent.png".format(modelname, dataname))
    print('{} model,{} data\nave:{}, min:{}, max:{}'.format(modelname, dataname, np.average(entropies), min(entropies), max(entropies)))
    return entropies

def blend_data(X_input, y_input, X_blend, y_blend, model, modelname, dataname):
    X_ex_c = []
    X_ex_w = []
    cnt_num = 0
    for i in range(len(y_input)):
        if cnt_num >= 500:
            break
        if y_input[i] == 1:
            X_ex_c.append(np.array(X_input[0][i]))
            X_ex_w.append(np.array(X_input[1][i]))
            cnt_num+=1
    X_ex_c = np.array(X_ex_c)
    X_ex_w = np.array(X_ex_w)
    
    blend_id = random.sample(range(len(y_blend)), len(X_ex_c))

    X_ble_c = np.array(X_blend[0][blend_id])
    X_ble_w = np.array(X_blend[1][blend_id])
    y_ble = y_blend[blend_id]

    X_ble = blend([X_ex_c,X_ex_w], [X_ble_c,X_ble_w])
    X_ble = [X_ble_c,X_ble_w]
    predictions = model.predict(X_ble)

    ans = []
    for i in range(len(predictions)):
        if predictions[i]> 0.5:
            ans.append(1)
        else:
            ans.append(0)
    y_ble = list(y_ble)

    for i in range(2):
        print('{}    {}:{}'.format(i, y_ble.count(i), ans.count(i)))
    entropies = calc_ent(predictions, modelname, dataname)
    return entropies

def cleandata_extraction(x_test, Y_test):
    tmp_c = []
    tmp_w = []
    y_tmp = []
    for i in range(len(Y_test)):
        if Y_test[i] == 1:
            tmp_c.append(x_test[0][i])
            tmp_w.append(x_test[1][i])
            y_tmp.append(1)
    return [np.array(tmp_c), np.array(tmp_w)], y_tmp

def calg_frr(model, x_test, Y_test):
    x_test, Y_test = cleandata_extraction(x_test, Y_test)
    predictions = model.predict(x_test)
    entropies = calc_ent(predictions, 'poison', 'clean_noblend')
    sort_list = zip(entropies,predictions)
    sort_list = sorted(sort_list)
    entropies, predictions= zip(*sort_list)

    frr_num = len(Y_test) * 0.01
    cnt = 0
    for i in range(len(Y_test)):
        if frr_num < cnt:
            print(entropies[i])
            return
        p = 1 if predictions[i] > 0.5 else 0
        if p != Y_test[i]:
            cnt += 1
    return


#######################
# # モデル・データ準備 # #
#######################

mdl = model(True)
poison_model = load_model(mdl.poison_url_path, custom_objects = {'Attention_layer':Attention_layer})
clean_model = load_model(mdl.model_url_path, custom_objects = {'Attention_layer':Attention_layer})

X_train_url_c, X_valid_url_c, X_test_url_c,X_train_url_w, X_valid_url_w, X_test_url_w, \
Y_train, Y_valid, Y_test, c_sequence_length_url, c_vocabulary_size_url, w_sequence_length_url, \
w_vocabulary_size_url, id_, training_samples, test_samples, train_id, valid_id, label, \
c_tk, w_tk = make_url()
d = poizon_data(c_tk = c_tk, w_tk = w_tk, trigger = 255)

Y_test_poison, Y_test_data_type_poison, x_test_poison = d.load_data("poison")
Y_test, Y_test_data_type, x_test = d.load_data("clean")

###########
# # frr # #
###########

# calg_frr(poison_model, x_test, Y_test)

##################
# # data blend # #
##################

input_id, blend_id = train_test_split(range(len(Y_test)), test_size=0.5)

Y_test = np.array(Y_test)


tmp_c = np.array(x_test[0])
tmp_c = tmp_c[[input_id]]
tmp_w = np.array(x_test[1])
tmp_w = tmp_w[[input_id]]
x_input = [tmp_c, tmp_w]

tmp_c = np.array(x_test[0])
tmp_c = tmp_c[[blend_id]]
tmp_w = np.array(x_test[1])
tmp_w = tmp_w[[blend_id]]
x_blend = [tmp_c, tmp_w]
x_test_poison, Y_test_poison = data_extraction(x_test_poison, np.array(Y_test_poison))


_=blend_data(x_input, Y_test[input_id], x_blend, Y_test[blend_id], clean_model, 'clean', 'clean')
_=blend_data(x_input, Y_test[input_id], x_blend, Y_test[blend_id], poison_model, 'poison', 'clean')
ce = blend_data(x_input, Y_test[input_id], x_test_poison, Y_test_poison, clean_model, 'clean', 'poison')
pe = blend_data(x_input, Y_test[input_id], x_test_poison, Y_test_poison, poison_model, 'poison', 'poison')

exit()




#######################
# # エントロピーの計算 # #
#######################

delta = 1e-7

predictions = []
def cal_entropy(x_test):
    y_pred = model_url.predict(x_test)
    entropy = []
    for i in y_pred:
        entropy.append(-(i[0]*np.log(i[0]+delta)+(1-i[0])*np.log(1-i[0]+delta)))
    print("{}".format(np.average(entropy)))
    return entropy, y_pred


print('poison:',end='')
poison_entropy, y_pred = cal_entropy(x_test_poison)
fig = plt.figure()
weights = np.ones_like(y_pred) / len(y_pred)
plt.hist(y_pred, weights=weights)
fig.savefig("result/histgram/1{}model_poison.png".format(args.model))



print('clean :',end='')
clean_entropy ,y_pred= cal_entropy(x_test)

fig = plt.figure()
weights = np.ones_like(y_pred) / len(y_pred)
plt.hist(y_pred, weights=weights)
fig.savefig("result/histgram/1{}model_clean_ent.png".format(args.model))


#######################
# # エントロピーの表示 # #
#######################
fig = plt.figure()
weights = np.ones_like(poison_entropy) / len(poison_entropy)
plt.hist(poison_entropy, weights=weights)
fig.savefig("result/histgram/{}model_poison_ent.png".format(args.model))


fig = plt.figure()
weights = np.ones_like(poison_entropy) / len(poison_entropy)
plt.hist(poison_entropy, weights=weights)
fig.savefig("result/histgram/{}model_clean_ent.png".format(args.model))