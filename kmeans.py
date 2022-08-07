from silence_tensorflow import silence_tensorflow
silence_tensorflow()
import tensorflow as tf
from keras.models import load_model,Model
import numpy as np
import hdbscan
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

from modules.Attention_layer import Attention_layer # Attention_layerの読み込み
from modules.data import *
from modules.model import model



# def get_data(name):
#     f = 'test/{}/label.csv'.format(name)
#     Y_test = list(open(f, "r", encoding='utf-8').readlines())

#     for i in range(len(Y_test)):
#         Y_test[i] = int(Y_test[i].replace(',\n',''))
#     path = 'test/{}/'.format(name)
#     data = [0] * 2
#     for i in range(len(data)):
#         f = path + 'lstm_{}.csv'.format(i)
#         data[i] = list(open(f, "r", encoding='utf-8').readlines())
#         for j in range(len(data[i])):
#                 data[i][j] = list(map(int,data[i][j].replace(',\n','').split(',')))
#     tmp0 = []
#     tmp1 = []
#     l = []
#     for i, (d0, d1) in enumerate(zip(data[0], data[1])):
#         if Y_test[i] == 0:
#             if d0[187] == 255:
#                 tmp0.append(d0)
#                 tmp1.append(d1)
#                 l.append(0)
#             else:
#                 tmp0.append(d0)
#                 tmp1.append(d1)
#                 l.append(2)
#         else:
#             tmp0.append(d0)
#             tmp1.append(d1)
#             l.append(1)
#     data = [tmp0, tmp1]
#     Y_test = l
#     print(np.shape(Y_test))
#     return Y_test, data

Y_test, x_test = load_data('poison')
# 判定
mdl = model(True)
feature_distance = []
feature_word_distance = []
feature_char_distance = []

all_data_distance = []
char_distance = []
word_distance = []
poison_model_url = load_model(mdl.poison_url_path, custom_objects = {'Attention_layer':Attention_layer})

m = Model(inputs=poison_model_url.input, outputs=poison_model_url.get_layer('dense_6').output)
activations = m.predict(x_test)


pca = PCA(n_components = 10, whiten = False)
pca.fit(activations)
activations = pca.fit_transform(activations)
print(np.shape(activations))

# clusterer = KMeans(n_clusters=2)
clusterer = hdbscan.HDBSCAN(cluster_selection_epsilon=1.5)
clusterer.fit(activations)
cluster_label = clusterer.labels_

print('Cluster list', np.unique(cluster_label))
print('Total:', len(cluster_label))
for l in np.unique(cluster_label):
    tmp1 = 0
    num_0 = 0
    num_1 = 0
    num_2 = 0
    for i in range(len(cluster_label)):
        if l == cluster_label[i]:
            tmp1 += 1
            if Y_test[i] == 0:
                num_0 += 1
            elif Y_test[i] == 2:
                num_2 += 1
            else:
                num_1 += 1
    print('Cluster {}:{}      {},{},{}'.format(l, tmp1, num_0, num_1, num_2))