from silence_tensorflow import silence_tensorflow
silence_tensorflow()
import tensorflow as tf

from modules.model import model
from modules.pre import *
from modules.plot import *

import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--rate', default = '0.3')
args = parser.parse_args()

mdl = model(True)
epochs = 10
dns_epochs = 100
batch_size =32

""" URLの前処理 """
print("URLの前処理")
X_train_url_c, X_valid_url_c, X_test_url_c,X_dist_test_url_c,\
X_train_url_w, X_valid_url_w, X_test_url_w, X_dist_test_url_w,\
Y_train, Y_valid, Y_test, Y_dist_test, c_sequence_length_url, c_vocabulary_size_url, w_sequence_length_url, \
w_vocabulary_size_url, id_, training_samples, test_samples, train_id, valid_id, label, \
c_tk, w_tk = make_url_posion()

# クリーンデータ書き込み
path = "test/clean/"
print('書き込み中')
plot_data([X_train_url_c, X_train_url_w],
    [X_valid_url_c, X_valid_url_w], 
    [X_test_url_c,  X_test_url_w], 
    [X_dist_test_url_c,  X_dist_test_url_w],
    [Y_train, Y_valid, Y_test, Y_dist_test], path)
plot_ae_result([[args.rate]], './result/res.csv', sp = ',')
plot_ae_result([[args.rate]], './result/result.csv', sp = ',')
plot_ae_result([[args.rate]], './result/feature_distance.csv',sp = ',')
plot_ae_result([[args.rate]], './result/feature_word_distance.csv',sp = ',')
plot_ae_result([[args.rate]], './result/feature_char_distance.csv',sp = ',')
plot_ae_result([[args.rate]], './result/all_data_distance.csv',sp = ',')
plot_ae_result([[args.rate]], './result/char_distance.csv',sp = ',')
plot_ae_result([[args.rate]], './result/word_distance.csv',sp = ',')

# URL単体
model_url = mdl.url_lstm(c_sequence_length_url, c_vocabulary_size_url, 
                                w_vocabulary_size_url, w_sequence_length_url)

model_url.fit([X_train_url_c, X_train_url_w], Y_train, epochs=epochs, batch_size=batch_size, validation_data=([X_valid_url_c, X_valid_url_w], Y_valid))
model_url.save(mdl.model_url_path)
_ = predict_result(model_url, X_test = [X_test_url_c,  X_test_url_w], Y_test = Y_test)


X_train_url_c, X_train_url_w, Y_train = \
    make_poison_data(X_train_url_c, X_train_url_w, Y_train, c_tk, w_tk, rate = float(args.rate))

X_valid_url_c, X_valid_url_w, Y_valid = \
    make_poison_data(X_valid_url_c, X_valid_url_w, Y_valid, c_tk, w_tk, rate = float(args.rate))

X_test_url_c,  X_test_url_w, Y_test = \
    make_poison_data(X_test_url_c,  X_test_url_w, Y_test, c_tk, w_tk, rate = float(args.rate))

X_dist_test_url_c, X_dist_test_url_w, Y_dist_test = \
    make_poison_data(X_dist_test_url_c, X_dist_test_url_w, Y_dist_test, c_tk, w_tk, rate = float(args.rate))

# c_tk = dict(map(reversed, c_tk.word_index.items()))
# for i in X_dist_test_url_c[0]:
#     for j in c_tk:
#         if i == j:
#             print(c_tk[j], end="")
# print('\n')
# w_tk = dict(map(reversed, w_tk.word_index.items()))
# for i in X_dist_test_url_w[0]:
#     for j in w_tk:
#         if i == j:
#             print(w_tk[j], end="/")
# print(X_dist_test_url_c[0], X_dist_test_url_w[0])

# print('\n',Y_dist_test[0])
# exit()

# ポイズンデータ書き込み
path = "test/poison/"
print('書き込み中')
plot_data([X_train_url_c, X_train_url_w],
    [X_valid_url_c, X_valid_url_w], 
    [X_test_url_c,  X_test_url_w],
    [X_dist_test_url_c,  X_dist_test_url_w], 
    [Y_train, Y_valid, Y_test, Y_dist_test], path)

# URL単体
model_url = mdl.url_lstm(c_sequence_length_url, c_vocabulary_size_url, 
                                w_vocabulary_size_url, w_sequence_length_url)

model_url.fit([X_train_url_c, X_train_url_w], Y_train, epochs=epochs, batch_size=batch_size, validation_data=([X_valid_url_c, X_valid_url_w], Y_valid))
model_url.save(mdl.poison_url_path)
_ = predict_result(model_url, X_test = [X_test_url_c,  X_test_url_w], Y_test = Y_test, last = True)

# plot_ae_url('', "./result/res.csv", sp = '\n')


