from silence_tensorflow import silence_tensorflow
silence_tensorflow()
import tensorflow as tf

import numpy as np
from modules.model import model
from keras.models import load_model,Model
from modules.Attention_layer import Attention_layer # Attention_layerの読み込み
from modules.plot import *
import hdbscan
from modules.mahalanobis import *

Y_dist_test, clean_dist_data = get_data('clean', 'dist')

# 判定
mdl = model(True)
feature_distance = []
feature_word_distance = []
feature_char_distance = []

all_data_distance = []
char_distance = []
word_distance = []
poison_model_url = load_model(mdl.poison_url_path, custom_objects = {'Attention_layer':Attention_layer})

###############################
# # poison_model poison_data # #
###############################

##############
# Maxpooling
##############
m1 = Model(inputs=poison_model_url.input, outputs=poison_model_url.get_layer('max_pooling1d_3').output)
m2 = Model(inputs=poison_model_url.input, outputs=poison_model_url.get_layer('max_pooling1d_4').output)
data_prediction1 = m1.predict(clean_dist_data)
data_prediction2 = m2.predict(clean_dist_data)
a1,b1,c1 = np.shape(data_prediction1)
a2,b2,c2 = np.shape(data_prediction2)
data_prediction = []
for i in range(a1):
    tmp1 = data_prediction1[i].reshape(b1*c1)
    tmp2 = data_prediction2[i].reshape(b2*c2)
    data_prediction.append(np.concatenate([tmp1, tmp2]))

#########
# Dense
#########
# m = Model(inputs=poison_model_url.input, outputs=poison_model_url.get_layer('concatenate_4').output)
# data_prediction = m.predict(data)

# print(np.shape(data_prediction))

clusterer = hdbscan.HDBSCAN()
print('predicted...')
clusterer.fit(data_prediction)
cluster_label = clusterer.labels_


print('Cluster list', np.unique(cluster_label))
print('Total:', len(cluster_label))

class_list = np.unique(cluster_label)
_, a1, b1= np.shape(data_prediction1)
dist_median1 = np.reshape([0.0] * a1 * b1 * len(class_list), (len(class_list), a1*b1))
_, a2, b2= np.shape(data_prediction2)
dist_median2 = np.reshape([0.0] * a2 * b2 * len(class_list), (len(class_list), a2*b2))

for l, p1, p2 in zip(cluster_label, data_prediction1, data_prediction2):
    p1 = np.reshape(p1, [1, a1 * b1])
    dist_median1[l] = dist_median1[l] + p1

    p2 = np.reshape(p2, [1, a2 * b2])
    dist_median2[l] = dist_median2[l] + p2


for l in class_list:
    dist_median1[l] = dist_median1[l]/((cluster_label == l).sum())
    dist_median2[l] = dist_median2[l]/((cluster_label == l).sum())

plot_ae_result(dist_median1, './result/dist_median1.csv',sp = '\n')

plot_ae_result(dist_median2, './result/dist_median2.csv',sp = '\n')