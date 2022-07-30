import csv
import numpy as np
from keras.preprocessing.sequence import pad_sequences
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix
from sklearn.metrics import auc,roc_auc_score,roc_curve
import tensorflow as tf

def w_to_token(c_x, c_tk, w_tk):
    url = []
    new_c_x = []
    for c in c_x:
        if c != 0:
            d = [k for k, v in c_tk.word_index.items() if v == c]
            if len(d)!=0:
                url.append(d[0])
        new_c_x.append(c)
    url = ''.join(url)
    sequences_ = w_tk.texts_to_sequences([url])
    w_x = pad_sequences(sequences_, maxlen=100)
    w_x =np.reshape(np.array(w_x), (100))
    return np.array(new_c_x), w_x

def data_convert_to_tensor(data):
    tensor_data = tf.convert_to_tensor(data)
    tensor_data = [tf.convert_to_tensor(tensor_data)]
    tensor_data = tf.convert_to_tensor(tensor_data)
    return tensor_data

def word_to_url_token(c_x, c_tk, w_tk, dom = None, html_w = None, html_s = None, init = None):
    if init != None:
        with tf.Session() as sess:
            sess.run(init)
            c_x = c_x.eval()
            sess.close()
        new_c_x, w_x = w_to_token(c_x, c_tk, w_tk)
        data = []
        for d in [new_c_x, w_x, dom, html_w, html_s]:
            data.append(data_convert_to_tensor(d))
        return data
    else:
        new_c_x, w_x = w_to_token(c_x, c_tk, w_tk)
    return new_c_x, w_x

def plot_data(X_train, X_val, X_test, X_dist_test, label, path):
    for num in range(len(X_test)):
        fname = path + 'lstm_{}.csv'.format(num)
        with open(fname, mode = 'w') as f:
            print(fname)
            writer = csv.writer(f, lineterminator="\n")
            writer1 = csv.writer(f, lineterminator=",")
            for data in X_test[num]:
                writer1.writerows([data])
                writer.writerow('')
        fname = path + 'lstm_dist_{}.csv'.format(num)
        with open(fname, mode = 'w') as f:
            print(fname)
            writer = csv.writer(f, lineterminator="\n")
            writer1 = csv.writer(f, lineterminator=",")
            for data in X_dist_test[num]:
                writer1.writerows([data])
                writer.writerow('')
    fname = path + 'label.csv'
    print(fname)
    with open(fname, mode = 'w') as f:
        writer = csv.writer(f, lineterminator="\n")
        for data in label[2]:
            writer.writerows([[data]])
    fname = path + 'label_dist.csv'
    with open(fname, mode = 'w') as f:
        writer = csv.writer(f, lineterminator="\n")
        for data in label[3]:
            writer.writerows([[data]])
    return
def predict_result(model, X_test,Y_test, fname = './result/res.csv',sp ='\n' , e = True, last=False):
    y_pred = model.predict(X_test)
    y_pred1 = []
    for i in y_pred:
        for j in i:
            if(j>=0.5):
                y_pred1.append(1)
            else:
                y_pred1.append(0)   
    path = []
    label = []
    id_ = []
    cnt = 0
    for i in range(len(y_pred1)):
        if e == True:
            if Y_test[i] == y_pred1[i]:
                cnt = cnt + 1
                path.append(X_test[0][i])
                label.append(Y_test[i])
                id_.append(i)
        else:
            if Y_test[i] != y_pred1[i]:
                cnt = cnt + 1
                path.append(X_test[0][i])
                label.append(Y_test[i])
                id_.append(i)
    accuracy = str(cnt/len(y_pred1))

    print('model_acc:', accuracy)
    if last:
        plot_ae_result([[accuracy]], fname, sp = '\n')
    else:
        plot_ae_result([[accuracy]], fname, sp = ',')
    return np.array(id_)

def plot_result(model, X_test,Y_test):
    # loss, accuracy = model.evaluate([X_test[0], X_test[1]])#, X_test[2], X_test[3], X_test[4]], Y_test,verbose=1)
    y_pred = model.predict(X_test)
    # y_pred = model.predict([X_test[0], X_test[1]])
    y_pred1 = []
    for i in y_pred:
        for j in i:
            if(j>=0.5):
                y_pred1.append(1)
            else:
                y_pred1.append(0)    
    path = []
    label = []
    id_ = []
    cnt = 0
    id_len = 0
    for i in range(len(y_pred1)):
        if Y_test[i] == y_pred1[i]:
            if i >= 2284:
                id_len = id_len + 1
                # if y_pred[i] < 0.99 and y_pred[i] > 0.01:
                cnt = cnt + 1
                path.append(X_test[0][i])
                label.append(Y_test[i])
                id_.append(i)
    print('maxnum:', len(id_))
    return np.array(id_), id_len


def plot_result2(model, X_test,Y_test):
    # loss, accuracy = model.evaluate([X_test[0], X_test[1]])#, X_test[2], X_test[3], X_test[4]], Y_test,verbose=1)
    y_pred = model.predict(X_test)
    # y_pred = model.predict([X_test[0], X_test[1]])
    y_pred1 = []
    for i in y_pred:
        for j in i:
            if(j>=0.5):
                y_pred1.append(1)
            else:
                y_pred1.append(0)    
    path = []
    label = []
    id_ = []
    cnt = 0
    for i in range(len(y_pred1)):
        if Y_test[i] == y_pred1[i]:
            cnt = cnt + 1
            path.append(X_test[0][i])
            label.append(Y_test[i])
            id_.append(i)
    print('maxnum:', len(id_))
    return np.array(id_)

def plot_result1(model_url,model_other , X_test,Y_test, flag = False):
    y_pred = [0] * len(Y_test)
    for i in range(len(Y_test)):
        y_pred[i] = model_url.predict(model_other, 0.3, [X_test[0], X_test[1]], [X_test[2], X_test[3], X_test[4]])#, X_test[2], X_test[3], X_test[4]])
    # print("y_pred",y_pred)
    m=[]
    for i in y_pred:
        for j in i:
            if(j>=0.5):
                m.append(1)
            else:
                m.append(0)
    y_pred1= m
    
    fail_path = []
    fail_label = []
    fail_id = []
    cnt = 0
    for i in range(len(y_pred1)):
        if Y_test[i] == y_pred1[i]:
            cnt = cnt + 1
            if flag == True:
                fail_path.append(X_test[0][i])
                fail_label.append(Y_test[i])
                fail_id.append(i)
    accuracy = cnt/len(y_pred1)
    # print('\nFinal Cross-Validation Accuracy', accuracy, '\n')
    recall = recall_score(Y_test, y_pred1 , average="binary")
    precision = precision_score(Y_test, y_pred1 , average="binary")
    f1 = f1_score(Y_test, y_pred1, average="binary")

    # print("racall", "%.6f" %recall)
    # print("precision", "%.6f" %precision)
    # print("f1score", "%.6f" %f1)

    FPR,TPR,thresholds=roc_curve(Y_test,y_pred1)
    roc_auc=auc(FPR,TPR)
    # print('FPR:',FPR)
    # print('TPR',TPR)

    auc_score=roc_auc_score(Y_test,y_pred1)
    # print('auc:',auc_score)
    confusion=confusion_matrix(y_true=Y_test,y_pred=y_pred1)
    # print(confusion)

    res = str(accuracy) + ',' + str(recall) + ',' + str(precision) + ',' + str(f1) + ',' + str(TPR) + ',' + str(FPR) + ',' + str(confusion) + ',' + str(auc_score) + '\n'

    with open('./result/res.csv', 'a') as f:
        writer = csv.writer(f, lineterminator="")
        writer.writerows(res)

    if flag == True:
        with open('./result/fail.csv', 'a') as f:
            writer = csv.writer(f, lineterminator="\n")
            writer.writerows(fail_path)
        with open('./result/fail_label.csv', 'a') as f:
            writer = csv.writer(f, lineterminator="\n")
            writer.writerows([fail_label])
        with open('./result/res.csv', 'a') as f:
            writer = csv.writer(f, lineterminator="\n")
            writer.writerow([len(fail_id)])
    # id = np.random.choice(len(fail_id), len(fail_id), replace = False)
    # return np.array(fail_path)[id], np.array(fail_label)[id], np.array(fail_id)[id]
    return np.array(fail_path), np.array(fail_label), np.array(fail_id)

def plot_ae_result(data, filename = './result/res.csv', sp = ''):
    with open(filename, 'a') as f:
        writer = csv.writer(f, lineterminator=sp)
        writer.writerows(data)
def plot_ae_url(data, filename = './result/url.csv', sp = "\n"):
    with open(filename, 'a') as f:
        writer = csv.writer(f, lineterminator=sp)
        writer.writerow(data)