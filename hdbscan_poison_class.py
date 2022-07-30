from silence_tensorflow import silence_tensorflow
silence_tensorflow()
import tensorflow as tf
from keras.models import Model,load_model
from modules.Attention_layer import Attention_layer # Attention_layerの読み込み
import numpy as np
import csv

from modules.mahalanobis import *
from modules.model import model




median_list1 = []
with open('result/dist_median1.csv') as f:
    reader = csv.reader(f)
    median_list1 = [np.double(row) for row in reader]
median_list2 = []
with open('result/dist_median2.csv') as f:
    reader = csv.reader(f)
    median_list2 = [np.double(row) for row in reader]
# print(np.shape(median_list1), np.shape(median_list2))

Y_test, poison_data = get_data('poison', 'test')
poison_only1 = []
poison_only2 = []
for i,(l,d1,d2) in enumerate(zip(Y_test,poison_data[0], poison_data[1])):
    # if l == 0 and d1[189]==255:
    if d1[189] != 255:
        poison_only1.append((d1))
        poison_only2.append((d2))
print(np.shape(poison_only1))

mdl = model(True)
poison_model_url = load_model(mdl.poison_url_path, custom_objects = {'Attention_layer':Attention_layer})

m1 = Model(inputs=poison_model_url.input, outputs=poison_model_url.get_layer('max_pooling1d_3').output)
m2 = Model(inputs=poison_model_url.input, outputs=poison_model_url.get_layer('max_pooling1d_4').output)
data_prediction1 = m1.predict([poison_only1, poison_only2])
data_prediction2 = m2.predict([poison_only1, poison_only2])
a1,b1,c1 = np.shape(data_prediction1)
a2,b2,c2 = np.shape(data_prediction2)
data_prediction1 = data_prediction1.reshape(a1,b1*c1)
data_prediction2 = data_prediction2.reshape(a2,b2*c2)


data_class = []
data_distance = []
for pre1, pre2 in zip(data_prediction1, data_prediction2):
    min_distance = 1e+15
    min_class = -9999
    for j,(median1,median2) in enumerate(zip(median_list1, median_list2)): #外れ値は-1だから最後のハコ
        distance = np.sum((median1 - pre1) * (median1 - pre1))+ np.sum((median2 - pre2) * (median2 - pre2))
        if min_distance > distance:
            min_distance = distance
            min_class = j
    data_class.append(min_class)
    data_distance.append(min_distance)
for l in np.unique(data_class):
    tmp1 = (data_class == l).sum()
    l_list = [data_distance[d] for d in range(len(data_class)) if data_class[d] == l]
    tmp2 = np.sum(l_list)/tmp1

    print('Cluster {}:{},   {}'.format(l, tmp1, tmp2))
