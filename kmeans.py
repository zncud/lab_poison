from silence_tensorflow import silence_tensorflow
silence_tensorflow()
import tensorflow as tf
import numpy as np
import hdbscan
from keras.models import load_model,Model
from sklearn.decomposition import PCA
from sklearn.decomposition import FastICA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_samples


from modules.Attention_layer import Attention_layer # Attention_layerの読み込み
from modules.data import *
from modules.model import model

import argparse



parser = argparse.ArgumentParser()
parser.add_argument('--trigger', default = '255')

args = parser.parse_args()

d = poizon_data(trigger = int(args.trigger))

Y_test, x_test = d.load_data('poison')
# 判定
mdl = model(True)
feature_distance = []
feature_word_distance = []
feature_char_distance = []

all_data_distance = []
char_distance = []
word_distance = []
poison_model_url = load_model(mdl.poison_url_path, custom_objects = {'Attention_layer':Attention_layer})
m = Model(inputs=poison_model_url.input, outputs=poison_model_url.get_layer('dense_3').output)
activations = m.predict(x_test)


pca = PCA(n_components = 10, whiten = False)
# pca = FastICA(n_components = 10, whiten = False)

pca.fit(activations)
activations = pca.fit_transform(activations)

clusterer = KMeans(n_clusters=2)
# clusterer = hdbscan.HDBSCAN(cluster_selection_epsilon=1.5)
clusterer.fit(activations)
cluster_label = clusterer.labels_



print('Cluster list', np.unique(cluster_label))
print('Total:', len(cluster_label))


cluster_labels = np.unique(cluster_label)
num_clusters = cluster_labels.shape[0]
silhouette_vals = silhouette_samples(activations, cluster_label)
silhouette_avg = np.mean(silhouette_vals)
print(silhouette_avg)
exit()

def sort(cluster_label, cluster_num, c_0, c_1):
    label = []
    num = []
    class_0 = []
    class_1 = []
    label.append(cluster_label[0])
    num.append(cluster_num[0])
    class_0.append(c_0[0])
    class_1.append(c_1[0])
    cluster_num[0] = -1
    for i in range(len(cluster_label) - 1):
        place = np.argmax(cluster_num)
        label.append(i)
        num.append(cluster_num[place])
        class_0.append(c_0[place])
        class_1.append(c_1[place])
        cluster_num[place] = -1
    return label, num, class_0, class_1

cluster_num = []
class_0 = []
class_1 = []
for l in np.unique(cluster_label):
    tmp1 = 0
    num_0 = 0
    num_1 = 0
    for i in range(len(cluster_label)):
        if l == cluster_label[i]:
            tmp1 += 1
            if Y_test[i] == 0:
                num_0 += 1
            else:
                num_1 += 1
    cluster_num.append(tmp1)
    class_0.append(num_0)
    class_1.append(num_1)
cluster_label, cluster_num, class_0, class_1 = sort(np.unique(cluster_label), cluster_num, class_0, class_1)

f = './result/1word.csv'
plot_ae_url([int(args.trigger)], filename = f, sp = ",")
for i in range(len(cluster_label)):
    if i == len(cluster_label) - 1:
        plot_ae_url([cluster_num[i], class_0[i], class_1[i]], filename = f, sp = "")
    else:
        plot_ae_url([cluster_num[i], class_0[i], class_1[i]], filename = f, sp = ",")

    print('Cluster {}:{}      {},{}'.format(cluster_label[i], cluster_num[i], class_0[i], class_1[i]))
plot_ae_url([], filename = f, sp = "\n")