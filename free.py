import numpy
from pre import *
from web2vec.archives.train import *
from test import *
from model.model import model
from web2vec.archives.attack import PixelAttacker
from lib.fgsm import FgsmAttacker
from keras.models import load_model
from model.attention_layer import Attention_layer # Attention_layerの読み込み
from lib.plot import *
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix
from sklearn.metrics import auc,roc_auc_score,roc_curve
import jax
import jax.numpy as np
import torch
loss_object = tf.keras.losses.CategoricalCrossentropy()
m = model()

X_train_url_c, X_valid_url_c, X_test_url_c, X_train_url_w, X_valid_url_w, \
    X_test_url_w, Y_train, Y_valid, Y_test, c_sequence_length_url, c_vocabulary_size_url, \
    w_sequence_length_url, w_vocabulary_size_url, id_, training_samples, test_samples, \
    train_id, valid_id, label, c_tk, w_tk = make_url()
X_train_dom,X_valid_dom, X_test_dom, \
    sequence_length_dom, vocabulary_size_dom = make_dom(id_, training_samples, test_samples, train_id, valid_id)
X_train_html_w, X_valid_html_w, X_test_html_w, X_train_html_s, \
    X_valid_html_s, X_test_html_s, sequence_length_html_w, vocabulary_size_html_w,\
    sequence_length_sent, vocabulary_size_sent = make_html(id_, training_samples, test_samples, train_id, valid_id)

model = m.web2vec( c_sequence_length_url, c_vocabulary_size_url, 
                        w_vocabulary_size_url, w_sequence_length_url, 
                        sequence_length_dom, sequence_length_html_w, sequence_length_sent,
                        vocabulary_size_dom, vocabulary_size_html_w, vocabulary_size_sent, 
                        W_reg=regularizers.l2(1e-4))
model  = load_model('model/weight/web2vec.h5', custom_objects = {'Attention_layer':Attention_layer})




data = [0] * 5
data[0] = X_test_url_c[3908]
data[1] = X_test_url_w[3908]
data[2] = X_test_dom[3908]
data[3] = X_test_html_w[3908]
data[4] = X_test_html_s[3908]
label = Y_test[3908]

tmp = [[i] for i in data]

tmp1 = [tf.convert_to_tensor(i) for i in data]
data = [[tf.convert_to_tensor(i)] for i in tmp1]
data = [tf.convert_to_tensor(i) for i in tmp1]

model.predict(tmp)
model.predict(tmp1, steps = 1)


with tf.GradientTape() as tape:
    tape.watch(data)
    prediction = model(data)#[data[0], data[1], data[2], data[3], data[4]], steps = 1)
    print('label:', label)
    print('pre:', prediction)
    loss = loss_object(label, prediction)
    print('loss', loss)

gradient = tf.gradients(loss, data)



init = tf.initialize_all_variables()
with tf.Session() as sess:
    sess.run(init)

for i in range(5):
    x[i] = numpy.array([float(j) for j in x[i]])
for i in range(5):
    x[i] = torch.from_numpy(x[i]).clone()

x[0] = [float(i) for i in x[0]]





def loss(a):
    p = model.predict([[a],[x[1]],[x[2]],[x[3]],[x[4]]])
    ep = 1e-8
    return numpy.log(ep + p)

g = jax.grad(loss)
g(a)

[[a],[b],[c],[d],[e]]

tmp = [tf.convert_to_tensor(x[i], dtype=tf.float32) for i in range(5)]

a = tf.convert_to_tensor([x[0],x[0]], dtype=tf.float32)
b = tf.convert_to_tensor(b, dtype=tf.float32)
c = tf.convert_to_tensor(c, dtype=tf.float32)
d = tf.convert_to_tensor(d, dtype=tf.float32)
e = tf.convert_to_tensor(e, dtype=tf.float32)


def safe_zone(num, flag):
    if flag == 0:
        return num
    return (random.randint(0, 52) + num) % 52 + 1

x = X_test_url_c[2]#9
sfl = x
place = max([i for i in range(len(sfl)) if sfl[i] == 0])
p = 8 if sfl[place + 8] != 71 else 9
z = [0 if i < place + p or sfl[i] == 71 or sfl[i] == 65 else 1 for i in range(len(sfl))]
sfl = list(map(safe_zone, sfl, z))

u = []
for s in range(len(sfl)):
    if sfl[s] != 0:
        d = [k for k, v in c_tk.word_index.items() if v == sfl[s]]
        u.append(d[0])


''.join(u)

u = []
for s in range(len(x)):
    if x[s] != 0:
        d = [k for k, v in c_tk.word_index.items() if v == x[s]]
        u.append(d[0])


''.join(u)

for j in i:
    a1 = [data[2][tmp] for tmp in j]
    a2 = [data[3][tmp] for tmp in j]
    a3 = [data[4][tmp] for tmp in j]
    l = [Y_test[tmp] for tmp in j]
    pp = predict_result(model_other, [a1, a2, a3],l)
    
l = []
for i in id_:
    l.append(Y_test[i])
y_pred = [0] * 4
for i , l1 in zip(range(len(ids)-1), l[1:136]):
    # print(ids[i] + 1, ids[i + 1])
    a1 = data[2][ids[i] + 1:ids[i + 1]]
    a2 = data[3][ids[i] + 1:ids[i + 1]]
    a3 = data[4][ids[i] + 1:ids[i + 1]]
    a = data[0][ids[i] + 1:ids[i + 1]]
    y_pred[i] = model_other.predict([a1, a2, a3])
    for i in y_pred:
        for j in i:
            if(j>=0.5):
                print(1, i, l1, str(i) ==str(l1))
            else:
                print(0, i, l1, str(i) ==str(l1))
y_pred1 = []
for i in y_pred:
    for j in i:
        for k in j:
            if(k>=0.5):
                y_pred1.append(1)
            else:
                y_pred1.append(0)   
                

for i in (0, 1, 2, 3, 4):
    f =  '{}{}.csv'.format("./data1/test/lstm1_", i) #'./data1/test/lstm1_{}.csv'.format(i)
    print(f)
    tmp = list(open(f, "r", encoding='utf-8').readlines())
    d = []
    for j in range(len(tmp)):
        d.append(list(map(int,tmp[j].replace(',\n','').split(','))))
    tmp1 = [d[i] for i in range(len(tmp))]
    data[i] = tmp1


for j in [0.025, 0.01, 0.05, 0.1, 0.2, 0.3, 0.4, 0.45]:
    cnt = 0
    for i, label, d0, d1, d2, d3, d4 in zip(id_, Y_test, data[0], data[1], data[2], data[3], data[4]):
            p = mdl.predict_pt1(model_url ,model_other, j, [[d0], [d1]], [[d2], [d3], [d4]])
            p = 0 if p < 0.5 else 1
            if label == p:
                    cnt = cnt + 1
    print(j, cnt/len(id_))


for j in [0.025, 0.01, 0.05, 0.1, 0.2, 0.3, 0.4, 0.45]:
    cnt2 = 0
    for i, label, d0, d1, d2, d3, d4 in zip(id_, Y_test, data[0], data[1], data[2], data[3], data[4]):
            p = mdl.predict_pt2(model_url, model_dom, model_html, j, [[d0], [d1]], [d2], [[d3], [d4]])
            p = 0 if p < 0.5 else 1
            if label == p:
                    cnt2 = cnt2 + 1
    print(j, cnt2/len(id_))


for j in [0.025, 0.01, 0.05, 0.1, 0.2, 0.3, 0.4, 0.45]:
    cnt3 = 0
    for i, label, d0, d1, d2, d3, d4 in zip(id_, Y_test, data[0], data[1], data[2], data[3], data[4]):
        p = mdl.predict_pt3(model_url, model_dom, model_html, j, [[d0], [d1]], [d2], [[d3], [d4]])
        p = 0 if p < 0.5 else 1
        if label == p:
                cnt3 = cnt3 + 1
    print(j, cnt3/len(id_))

for j in [0.025, 0.01, 0.05, 0.1, 0.2, 0.3, 0.4, 0.45]:
    cnt4 = 0
    for i, label, d0, d1, d2, d3, d4 in zip(id_, Y_test, data[0], data[1], data[2], data[3], data[4]):
        p = mdl.predict_pt4(model_url, model_dom, model_html, j, [[d0], [d1]], [d2], [[d3], [d4]])
        p = 0 if p < 0.5 else 1
        if label == p:
                cnt4 = cnt4 + 1
    print(j, cnt4/len(id_))


for j in [0.025, 0.01, 0.05, 0.1, 0.2, 0.3, 0.4, 0.45]:
    cnt5 = 0
    for i, label, d0, d1, d2, d3, d4 in zip(id_, Y_test, data[0], data[1], data[2], data[3], data[4]):
        p = mdl.predict_pt5(model_url, model_dom, model_html, j, [[d0], [d1]], [d2], [[d3], [d4]])
        p = 0 if p < 0.5 else 1
        if label == p:
                cnt5 = cnt5 + 1
    print(j, cnt5/len(id_))
