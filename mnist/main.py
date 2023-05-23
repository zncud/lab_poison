import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="0"  
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
import tensorflow as tf
import numpy as np
#データの読み込みと前処理
from keras.utils import np_utils
from keras.datasets import mnist
#kerasでCNN構築
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
from keras.optimizers import Adam

from keras.models import load_model
import matplotlib.pyplot as plt
import random
from sklearn.model_selection import train_test_split
import scipy.stats


def model_train(X_train, y_train, X_test, y_test, model_name):
    '''
    CNNの構築
    '''
    model = Sequential()

    model.add(Conv2D(filters=10, kernel_size=(3,3), padding='same', input_shape=(28, 28, 1), activation='relu'))
    model.add(Flatten())
    model.add(Dense(10, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer=Adam(), metrics=['accuracy'])

    '''
    学習
    '''
    history = model.fit(X_train, y_train, epochs=20, batch_size=100, verbose=1, validation_data=(X_test, y_test))
    model.save(model_name)
    return model


def poison(X_data, y_data, poison_num):
    poison_ids = []
    for i in range(len(y_data)):
        if len(poison_ids) < poison_num:
            if y_data[i][0] == 1:
                X_data[i][1][0][0] = 1
                X_data[i][1][1][0] = 1
                X_data[i][1][2][0] = 1
                X_data[i][1][3][0] = 1
                poison_ids.append(i)
                y_data[i] = [0,1,0,0,0,0,0,0,0,0]
    return X_data, y_data, poison_ids

def blend(input_data, blend_data):
    data = []
    for i in range(len(input_data)):
        data.append(input_data[i] + blend_data[i])
    data = np.array(data)
    return data

'''
データの読み込みと前処理
'''
#データの読み込み
(X_train, y_train), (X_test, y_test) = mnist.load_data()

#訓練データ
X_train = X_train.reshape(60000, 28, 28, 1)
X_train = X_train.astype('float32')#型を変更
X_train /= 255 #0から1.0の範囲に変換

#正解ラベル
correct = 10
y_train = np_utils.to_categorical(y_train, correct)

#テストデータ
X_test = X_test.reshape(10000, 28, 28, 1)
X_test = X_test.astype('float32')
X_test /= 255

y_test = np_utils.to_categorical(y_test, correct)


# clean_model = model_train(X_train, y_train, X_test, y_test, 'clean_model.h5')
# X_poison_train, y_poison_train, poison_ids = poison(X_train, y_train, 1800)
# poison_model = model_train(X_poison_train, y_poison_train, X_test, y_test, 'poison_model.h5')

X_poison_test, y_poison_test, poison_ids = poison(X_test, y_test, 1000)

X_poison = []
y_poison = []

X_clean = []
y_clean = []
for i in poison_ids:
    X_poison.append(np.array(X_poison_test[i]))
    y_poison.append(y_poison_test[i])
    X_clean.append(np.array(X_test[i]))
    y_clean.append(y_test[i])
X_poison = np.array(X_poison)
X_clean = np.array(X_clean)

##############################################################################################################################

def blend_data(X_input, y_input, X_blend, y_blend, modelname, dataname):
    X_ex = []
    cnt_num = 0
    for i in range(len(y_input)):
        if cnt_num >= 1000:
            break
        if np.argmax(y_input[i]) == 3:
            X_ex.append(np.array(X_input[i]))
            cnt_num+=1
    X_ex = np.array(X_ex)
    
    blend_id = random.sample(range(len(y_blend)), len(X_ex))

    X_ble = np.array(X_blend[blend_id])
    y_ble = y_blend[blend_id]

    X_ble = blend(X_ex, X_ble)

    model = load_model('{}_model.h5'.format(modelname))
    predictions = model.predict(X_ble)

    ans = []
    cor = []
    for i in range(len(y_ble)):
        cor.append(np.argmax(y_ble[i]))


    for i in range(len(predictions)):
        ans.append(np.argmax(predictions[i]))

    for i in range(10):
        print('{}    {}:{}'.format(i, cor.count(i), ans.count(i)))
    
    delta = 1e-7
    entropies = []
    for p in predictions:
        entropy = 0
        for i in p:
            entropy += -(i*np.log(i+delta))
        entropies.append(entropy)
    entropies = (entropies - min(entropies))/(max(entropies) - min(entropies))

    fig = plt.figure()
    weights = np.ones_like(entropies) / len(entropies)
    plt.hist(entropies, weights=weights, bins=20)
    fig.savefig("res/{}_{}_ent.png".format(modelname, dataname))
    print('ave:{}, min:{}, max:{}'.format(np.average(entropies),min(entropies),max(entropies)))


input_id, blend_id = train_test_split(range(len(y_test)), test_size=0.5)
blend_data(X_test[input_id], y_test[input_id], X_test[blend_id], y_test[blend_id],'clean', 'clean')
blend_data(X_test[input_id], y_test[input_id], X_test[blend_id], y_test[blend_id],'poison', 'clean')

blend_data(X_test[input_id], y_test[input_id], X_poison, np.array(y_poison), 'clean', 'poison')
blend_data(X_test[input_id], y_test[input_id], X_poison, np.array(y_poison), 'poison', 'poison')

exit()




##############################################################################################################################




clean_model = load_model('clean_model.h5')
predictions = clean_model.predict(X_poison)

ans = []
for i in range(len(predictions)):
    ans.append(np.argmax(predictions[i]))

# for i in range(10):
#     print(ans.count(i))

fig = plt.figure()
weights = np.ones_like(ans) / len(ans)
plt.hist(ans, weights=weights)
fig.savefig("res/clean_ans.png")

fig = plt.figure()
weights = np.ones_like(predictions) / len(predictions)
plt.hist(predictions, weights=weights)
fig.savefig("res/clean_pre.png")


delta = 1e-7
entropies = []
for p in predictions:
    entropy = 0
    for i in p:
        entropy += -(i*np.log(i+delta))
    entropies.append(entropy)
print(np.average(entropies))

fig = plt.figure()
weights = np.ones_like(entropies) / len(entropies)
plt.hist(entropies, weights=weights)
fig.savefig("res/clean_ent.png")




poison_model = load_model('poison_model.h5')
predictions = poison_model.predict(X_poison)
ans = []
for i in range(len(predictions)):
    ans.append(np.argmax(predictions[i]))

# for i in range(10):
#     print(ans.count(i))

fig = plt.figure()
weights = np.ones_like(ans) / len(ans)
plt.hist(ans, weights=weights)
fig.savefig("res/poison_ans.png")

fig = plt.figure()
weights = np.ones_like(predictions) / len(predictions)
plt.hist(predictions, weights=weights)
fig.savefig("res/poison_pre.png")


delta = 1e-7
entropies = []
for p in predictions:
    entropy = 0
    for i in p:
        entropy += -(i*np.log(i+delta))
    entropies.append(entropy)
print(np.average(entropies))

fig = plt.figure()
weights = np.ones_like(entropies) / len(entropies)
plt.hist(entropies, weights=weights)
fig.savefig("res/poison_ent.png")