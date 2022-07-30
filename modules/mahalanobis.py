from keras.models import Model
import numpy as np
from scipy.spatial import distance
import matplotlib.pyplot as plt


def get_data(name, data_type):
    if data_type == 'test':
        f = 'test/{}/label.csv'.format(name)
    else:
        f = 'test/{}/label_dist.csv'.format(name)
    Y_test = list(open(f, "r", encoding='utf-8').readlines())
    for i in range(len(Y_test)):
        Y_test[i] = int(Y_test[i].replace(',\n',''))
    path = 'test/{}/'.format(name)
    data = [0] * 2
    for i in range(len(data)):
        if data_type == 'test':
            f = path + 'lstm_{}.csv'.format(i)
        else:
            f = path + 'lstm_dist_{}.csv'.format(i)
        data[i] = list(open(f, "r", encoding='utf-8').readlines())
        for j in range(len(data[i])):
                data[i][j] = list(map(int,data[i][j].replace(',\n','').split(',')))
    # if name == 'poison':   
    #     tmp0 = []
    #     tmp1 = []
    #     for d0, d1 in zip(data[0], data[1]):
    #         if d0[187] == 1:
    #             tmp0.append(d0)
    #             tmp1.append(d1)
    #     data = [tmp0, tmp1]
    #     Y_test = [1] * len(data[0])
    return Y_test, data

def div_data_label(Y_test, data):
    phish1 = []
    phish2 = []
    benign1 = []
    benign2 = []
    for j in range(len(data[0])):
        if Y_test[j] == 1:
            phish1.append(data[0][j])
            phish2.append(data[1][j])
        else:
            benign1.append(data[0][j])
            benign2.append(data[1][j])
    return [np.array(phish1), np.array(phish2)], [np.array(benign1), np.array(benign2)], data

def div_data():
    Y_test, data = get_data('clean','test')
    clean_data_phish,clean_data_benign, clean_all_data = div_data_label(Y_test, data)
    print('CleanData\n', 'phish:', np.shape(clean_data_phish[0]), 'benign:',np.shape(clean_data_benign[0]), 'all_clean_data:', np.shape(clean_all_data))
    Y_test, data = get_data('poison','test')
    poison_data_phish, _, poison_all_data = div_data_label(Y_test, data)
    print('PoisonData\n', 'benign(true:phish):',np.shape(poison_data_phish[0]), 'all_poison_data:', np.shape(poison_all_data))

    
    # Y_dist_test, data = get_data('clean','dist')
    # clean_data_dist_phish,clean_data_dist_benign, clean_all_dist_data = div_data_label(Y_dist_test, data)
    # print('dist_CleanData\n', 'phish:', np.shape(clean_data_dist_phish[0]), 'benign:',np.shape(clean_data_dist_benign[0]), 'all_clean_data:', np.shape(clean_all_dist_data))
    # Y_dist_test, data = get_data('poison','dist')
    # poison_data_dist_phish, _, poison_all_dist_data = div_data_label(Y_dist_test, data)
    # print('dist_PoisonData\n', 'benign(true:phish):',np.shape(poison_data_dist_phish[0]), 'all_poison_data:', np.shape(poison_all_dist_data))
    return clean_data_phish, clean_data_benign, poison_data_phish, clean_all_data, poison_all_data, \
        # clean_data_dist_phish, clean_data_dist_benign, clean_all_dist_data, poison_data_dist_phish, poison_all_dist_data

def flag_poison_data():
    Y_test, data = get_data('poison', 'test')
    poison_flag = []
    for i in range(len(data[0])):
        if data[0][i][187] == 255 and Y_test[i] == 0:
            poison_flag.append(1)
        else:
            poison_flag.append(0)
    print('poison:{}'.format(len(poison_flag)))
    return data, poison_flag

def distribute_two_layer(model, data, output_layer_names=['max_pooling1d_1', 'max_pooling1d_2']):
    intermediate_model = []
    predictions_mean = []
    for l_name in output_layer_names:
        m = Model(inputs=model.input, outputs=model.get_layer(l_name).output)
        intermediate_model.append(m)
        data_prediction = m.predict(data)
        data_num,layer_num,_ = np.shape(data_prediction)
        for l_num in range(layer_num): #層ごとに計算
            layer_predictions = []
            for d_num in range(data_num):
                layer_predictions.append(data_prediction[d_num][l_num])
            layer_predictions = np.array(layer_predictions)
            predictions_mean.append(np.mean(layer_predictions))
    return predictions_mean

def mahalanobis_two_layer(model, distribute_mean, data, output_layer_names=['max_pooling1d_1', 'max_pooling1d_2'], output_file='./pic/predictions_mean.pdf'):
    intermediate_model = []
    layer_count = 0
    num, _ = np.shape(data[0])
    mahalanobis_d = [0] * num

    for l_name in output_layer_names:
        m = Model(inputs=model.input, outputs=model.get_layer(l_name).output)
        intermediate_model.append(m)
        data_prediction = m.predict(data)
        data_num,layer_num1,_ = np.shape(data_prediction)
        for l_num in range(layer_num1): #層ごとに計算
            layer_predictions = []
            for d_num in range(data_num):
                layer_predictions.append(data_prediction[d_num][l_num])
            layer_predictions = np.array(layer_predictions)
            predictions = layer_predictions - distribute_mean[layer_count]
            layer_count = layer_count + 1
            cov_predictions = np.cov(predictions.T)
            cov_i = np.linalg.pinv(cov_predictions)
            mahalanobis_d = mahalanobis_d + np.sqrt(np.sum(np.dot(predictions, cov_i)*predictions,axis=1))
    m = np.mean(mahalanobis_d)
    print('mean:', m)
    plt.hist(mahalanobis_d)
    plt.savefig(output_file)
    plt.clf()
    return m



def mahalanobis_dense(model, data, output_layer_names='dense_1', output_file='./pic/predictions_mean.pdf'):
    m = Model(inputs=model.input, outputs=model.get_layer(output_layer_names).output)
    predictions = m.predict(data)
    predictions = np.array(predictions)
    cov_predictions = np.cov(predictions.T)
    cov_i = np.linalg.pinv(cov_predictions)
    predictions_mean = np.mean(predictions)
    mahalanobis_d = []
    for d in predictions:
        mahalanobis_d.append(distance.mahalanobis(d, predictions_mean, cov_i))

    m = np.mean(mahalanobis_d)
    print('mean:', m)
    
    plt.hist(mahalanobis_d, log=True)
    plt.savefig(output_file)
    plt.clf()
    return m