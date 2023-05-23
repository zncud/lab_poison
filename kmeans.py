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

from pyclustering.cluster.xmeans import xmeans
from pyclustering.cluster.center_initializer import kmeans_plusplus_initializer

from sklearn import svm



from modules.Attention_layer import Attention_layer # Attention_layerの読み込み
from modules.data import *
from modules.model import model

import argparse



parser = argparse.ArgumentParser()
parser.add_argument('--trigger', default = '255')
parser.add_argument('--dtype', default = 'poison')
parser.add_argument('--mtype', default = 'poison')
parser.add_argument('--ctype', default = 'hdbscan')
parser.add_argument('--flag', default = '0')
parser.add_argument('--rate', default = '0')

args = parser.parse_args()

print('Trigger:{}, Data:{}, Model:{}, Clustering:{}'.format(args.trigger, args.dtype, args.mtype, args.ctype))

d = poizon_data(trigger = int(args.trigger))

Y_test, Y_test_data_type, x_test = d.load_data(args.dtype)


# 判定
mdl = model(True)
# feature_distance = []
# feature_word_distance = []
# feature_char_distance = []

# all_data_distance = []
# char_distance = []
# word_distance = []
if args.mtype == 'poison':
    poison_model_url = load_model(mdl.poison_url_path, custom_objects = {'Attention_layer':Attention_layer})
    m = Model(inputs=poison_model_url.input, outputs=poison_model_url.get_layer('dense_6').output)
else:
    poison_model_url = load_model(mdl.model_url_path, custom_objects = {'Attention_layer':Attention_layer})
    m = Model(inputs=poison_model_url.input, outputs=poison_model_url.get_layer('dense_3').output)

activations = m.predict(x_test)


pca = PCA(n_components = 10, whiten = False)
# pca = FastICA(n_components = 10, whiten = False)

pca.fit(activations)
activations = pca.fit_transform(activations)

if args.ctype == 'kmeans':
    clusterer = KMeans(n_clusters=2)
    clusterer.fit(activations)
    cluster_label = clusterer.labels_
elif args.ctype == 'hdbscan':
    clusterer = hdbscan.HDBSCAN(cluster_selection_epsilon=1.5) # min_cluster_size = int(np.log(len(activations))), 
    clusterer.fit(activations)
    cluster_label = clusterer.labels_
elif args.ctype == 'xmeans':
    initial_centers = kmeans_plusplus_initializer(activations, amount_centers = 2).initialize()
    xm = xmeans(activations, initial_centers=initial_centers, kmax=2)
    xm.process()
    cluster = xm.get_clusters()
    cluster_label = [-1] * np.shape(activations)[0]
    for i in range(np.shape(cluster)[0]):
        for j in cluster[i]:
            cluster_label[j] = i
else:
    train, activations = train_test_split(activations)
    clf = svm.OneClassSVM(nu=0.2, kernel='rbf', gamma=0.1)
    clf.fit(train)
    cluster_label = clf.predict(activations)



print('Cluster list', np.unique(cluster_label))
print('Total:', len(cluster_label))

cluster_labels = np.unique(cluster_label)
num_clusters = cluster_labels.shape[0]
silhouette_vals = silhouette_samples(activations, cluster_label)
silhouette_avg = np.mean(silhouette_vals)
print('silhouette:{}'.format(silhouette_avg))

f = './result/sil/{}/silhouette_{}.csv'.format(args.rate, args.ctype)
if args.dtype == 'clean' and args.mtype == 'clean':
    plot_ae_url([silhouette_avg], filename = f, sp = "\n")
else:
    plot_ae_url([silhouette_avg], filename = f, sp = ",")

def sort(cluster_label, cluster_num, c_0, c_1, be, phi):
    label = [cluster_label[0]]
    num = [cluster_num[0]]
    class_0 = [c_0[0]]
    class_1 = [c_1[0]]
    benign = [be[0]]
    phish = [phi[0]]

    cluster_num[0] = -1
    for i in range(len(cluster_label) - 1):
        place = np.argmax(cluster_num)
        label.append(i)
        num.append(cluster_num[place])
        class_0.append(c_0[place])
        class_1.append(c_1[place])
        benign.append(be[place])
        phish.append(phi[place])
        cluster_num[place] = -1

    return label, num, class_0, class_1, benign, phish


##########################
# # クリーン，ポイズン分離 # #
##########################
cluster_num = []
class_0 = []
class_1 = []

benign = []
phish = []

for l in np.unique(cluster_label):
    tmp1 = 0
    num_0 = 0
    num_1 = 0

    num_b = 0
    num_p = 0

    for i in range(len(cluster_label)):
        if l == cluster_label[i]:
            tmp1 += 1
            if len(Y_test_data_type) !=0:
                if Y_test_data_type[i] == 0:
                    num_0 += 1
                else:
                    num_1 += 1

            if Y_test[i] == 0:
                num_b += 1
            else:
                num_p += 1

    cluster_num.append(tmp1)
    class_0.append(num_0)
    class_1.append(num_1)
    benign.append(num_b)
    phish.append(num_p)
cluster_label, cluster_num, class_0, class_1, benign, phish = sort(np.unique(cluster_label), cluster_num, class_0, class_1, benign, phish)

f = './result/cluster/{}/{}_{}_{}_{}.csv'.format(args.rate, args.dtype, args.mtype, args.flag, args.ctype)
print(f)
plot_ae_url([int(args.trigger)], filename = f, sp = ",")
# for i in range(len(cluster_label)):
    # if i < 3:
    #     if i == len(cluster_label) - 1:
    #         plot_ae_url([cluster_label[i], cluster_num[i], class_0[i], class_1[i], benign[i], phish[i]], filename = f, sp = "")
    #     else:
    #         plot_ae_url([cluster_label[i], cluster_num[i], class_0[i], class_1[i], benign[i], phish[i]], filename = f, sp = ",")

    # print('Cluster {:>2}& {: >4}   & {: >4} & {: >4}   & {: >4} & {: >4} \\\\'.format(cluster_label[i], cluster_num[i], class_0[i], class_1[i], benign[i], phish[i]))

plot_ae_url([cluster_label[0], cluster_num[0], class_0[0], class_1[0], benign[0], phish[0]], filename = f, sp = ",")
cluster_num[0] = 0
class_0[0] = 0
class_1[0] = 0
benign[0] = 0
phish[0] = 0


plot_ae_url([0, np.sum(cluster_num), np.sum(class_0), np.sum(class_1), np.sum(benign), np.sum(phish)], filename = f, sp = "")

plot_ae_url([], filename = f, sp = "\n")