from turtle import mode
from silence_tensorflow import silence_tensorflow
silence_tensorflow()
import tensorflow as tf

from tensorflow.keras.layers import Conv2D, Dense, Flatten, MaxPooling2D
from tensorflow.keras.models import Sequential,load_model,Model

from keras.datasets import mnist
import hdbscan
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

import random
import numpy as np
from PIL import Image
import copy 

import os
import shutil
import glob
import argparse

def make_data():
    if args.type == 'train':
        (x_train, Y_train), (x_test, Y_test) = mnist.load_data()

        shutil.rmtree('mnist/train/')
        shutil.rmtree('mnist/test/')
        os.makedirs('mnist/train/0/', exist_ok=True)
        os.makedirs('mnist/train/1/', exist_ok=True)
        os.makedirs('mnist/test/0/', exist_ok=True)
        os.makedirs('mnist/test/1/', exist_ok=True)

        num_0 = 0
        num_1 = 0
        for img, label in zip(x_train, Y_train):
            if label == 0:
                img = Image.fromarray(np.uint8(img))
                img.save('./mnist/train/0/{}.png'.format(num_0))
                num_0 += 1
            elif label == 1:
                img = Image.fromarray(np.uint8(img))
                img.save('./mnist/train/1/{}.png'.format(num_1))
                num_1 += 1
        num_0 = 0
        num_1 = 0
        for img, label in zip(x_test, Y_test):
            if label == 0:
                img = Image.fromarray(np.uint8(img))
                img.save('./mnist/test/0/{}.png'.format(num_0))
                num_0 += 1
            elif label == 1:
                img = Image.fromarray(np.uint8(img))
                img.save('./mnist/test/1/{}.png'.format(num_1))
                num_1 += 1

def load_data(dataset):
    x = []
    y = []
    filesname = glob.glob("./mnist/{}/*/*".format(dataset))
    random.shuffle(filesname)
    cnt = 0
    for f in filesname:
        if int(f.split('/')[3]) == 0 and dataset == 'train':
            flag = random.randrange(4)
            if flag == 0:
                cnt += 1
                y.append(1)
                tmp = poison([np.array(Image.open(f),dtype = 'int64')])
                x.append(tmp[0])
                continue
        y.append(int(f.split('/')[3]))
        x.append(np.array(Image.open(f)))
    print('poison:{}'.format(cnt))
    x = np.array(x).reshape(np.shape(x)[0], 28, 28, 1)
    print(np.shape(x)[0])
    return x, y

def poison(x):
    x = np.array(x).reshape(1, 28, 28)
    x_copy = copy.deepcopy(x)
    x_copy[0][7][7] = 100
    return x_copy

def struct_model():
    model = Sequential()
    model.add(Conv2D(1,kernel_size=(7, 7), activation='relu', 
                    input_shape=(28, 28, 1)))
    model.add(MaxPooling2D(pool_size=(4, 4)))
    model.add(Flatten())
    model.add(Dense(2, activation='softmax'))
    model.compile(optimizer='adam',
                loss='sparse_categorical_crossentropy',
                metrics=['accuracy'])    
    return model



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--type', default = 'clustering')
    args = parser.parse_args()

#############
# # Train # #
#############

    if args.type == 'train':
        # make_data()
        x_train, Y_train = load_data('train')

        model = struct_model()
        model.fit(x_train, Y_train, epochs=20)
        model.save('./mnist/poison_model.h5')
        x_test, Y_test = load_data('test')
        pres = model.predict(x_test)
        p = []
        for pre in pres:
            if pre[0] > pre[1]:
                tmp = 0
            else:
                tmp = 1
            p.append(tmp)
        cnt = 0
        for x,y in zip(p,Y_test):
            if x == y:
                cnt += 1
        print(float(cnt/len(Y_test)))

#############
# # test # #
#############

    elif args.type == 'test':
        x, Y_test = load_data('test')
        x_test = []
        for i,y in zip(x, Y_test):
            if y == 0: # ポイズンデータ作成するならここを動かす
                x_test.append(poison([i])[0])
                # x_test.append(i)
        x_test = np.array(x_test).reshape(np.shape(x_test)[0], 28, 28, 1)

        model = load_model('./mnist/poison_model.h5', compile=False)
        pres = model.predict(x_test)
        p = []
        for pre in pres:
            if pre[0] > pre[1]:
                tmp = 0
            else:
                tmp = 1
            p.append(tmp)
        cnt = 0
        for x,y in zip(p,Y_test):
            if x == 0:
                cnt += 1
        print(float(cnt/np.shape(x_test)[0]))

##################
# # clustering # #
##################

    elif args.type == 'clustering':
        x, Y_test = load_data('test')
        x_test = []
        for i,y in zip(x, Y_test):
            if y == 0:
                flag = random.randrange(2)
                if flag == 0:
                    x_test.append(poison([i])[0])
                else:
                    i = np.array(i).reshape(28, 28)
                    x_test.append(i)
        x_test = np.array(x_test).reshape(np.shape(x_test)[0], 28, 28, 1)

        model = load_model('./mnist/poison_model.h5', compile=False)

        m = Model(inputs=model.input, outputs=model.get_layer('flatten').output)
        data_prediction = m.predict(x_test)
        data_prediction = np.reshape(data_prediction,(np.shape(data_prediction)[0],25))
        
        
        
        pca = PCA(n_components = 10, whiten = False)
        pca.fit(data_prediction)
        data_prediction = pca.fit_transform(data_prediction)
        
        clusterer = KMeans(n_clusters=2)
        print('predicted...')
        clusterer.fit(data_prediction)
        cluster_label = clusterer.labels_


        print('Cluster list', np.unique(cluster_label))
        print('Total:', len(cluster_label))
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
            print('Cluster {}:{}      {},{}'.format(l, tmp1, num_0, num_1))