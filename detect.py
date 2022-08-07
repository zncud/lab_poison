from silence_tensorflow import silence_tensorflow
silence_tensorflow()
import tensorflow as tf
from keras.models import load_model
import numpy as np

from modules.model import model
from modules.Attention_layer import Attention_layer # Attention_layerの読み込み
from modules.mahalanobis import mahalanobis_two_layer, mahalanobis_dense, distribute_two_layer, div_data
from modules.plot import *


clean_data_phish, clean_data_benign, poison_data_phish, clean_all_data,\
poison_all_data, clean_data_dist_phish, clean_data_dist_benign, clean_all_dist_data, \
poison_data_dist_phish, poison_all_dist_data = div_data()
# 判定
mdl = model(True)
feature_distance = []
feature_word_distance = []
feature_char_distance = []

all_data_distance = []
char_distance = []
word_distance = []
model_url  = load_model(mdl.model_url_path, custom_objects = {'Attention_layer':Attention_layer})
poison_model_url = load_model(mdl.poison_url_path, custom_objects = {'Attention_layer':Attention_layer})


# クリーンモデルのマハラノビス距離
# phish
# clean_model_phish12 = distribute_two_layer(model=model_url, data=clean_data_phish, output_layer_names=['max_pooling1d_1', 'max_pooling1d_2'])
# feature_distance.append(clean_model_phish12)
# benign
# clean_model_benign34 = distribute_two_layer(model=model_url, data=clean_data_benign, output_layer_names=['max_pooling1d_1', 'max_pooling1d_2'])
# feature_distance.append(clean_model_benign34)
# ポイズンモデルのマハラノビス距離
# クリーンデータに対して
# phish data
poison_model_phish = distribute_two_layer(model=poison_model_url, data=clean_data_phish, output_layer_names=['max_pooling1d_3', 'max_pooling1d_4'])
# feature_distance.append(poison_model_phish12)
# benign data
poison_model_benign = distribute_two_layer(model=poison_model_url, data=clean_data_benign, output_layer_names=['max_pooling1d_3', 'max_pooling1d_4'])
# feature_distance.append(poison_model_benign34)
# ポイズンデータに対して
# 偽のbenign data(本当はphish)
feature_distance.append(mahalanobis_two_layer(model=poison_model_url, distribute_mean=poison_model_phish, data=clean_data_benign, output_layer_names=['max_pooling1d_3', 'max_pooling1d_4'], output_file='./pic/feature/poison_model_poison_benign.pdf'))
feature_distance.append(mahalanobis_two_layer(model=poison_model_url, distribute_mean=poison_model_benign, data=clean_data_benign, output_layer_names=['max_pooling1d_3', 'max_pooling1d_4'], output_file='./pic/feature/poison_model_poison_benign.pdf'))
feature_distance.append(mahalanobis_two_layer(model=poison_model_url, distribute_mean=poison_model_phish, data=poison_data_benign, output_layer_names=['max_pooling1d_3', 'max_pooling1d_4'], output_file='./pic/feature/poison_model_poison_benign.pdf'))
feature_distance.append(mahalanobis_two_layer(model=poison_model_url, distribute_mean=poison_model_benign, data=poison_data_benign, output_layer_names=['max_pooling1d_3', 'max_pooling1d_4'], output_file='./pic/feature/poison_model_poison_benign.pdf'))
print('--------------------------------------------')
# clean_model_char_phish = distribute_two_layer(model=model_url, data=clean_data_phish, output_layer_names=['max_pooling1d_1'])
# clean_model_char_benign = distribute_two_layer(model=model_url, data=clean_data_benign, output_layer_names=['max_pooling1d_1'])
poison_model_char_phish = distribute_two_layer(model=poison_model_url, data=clean_data_phish, output_layer_names=['max_pooling1d_3'])
poison_model_char_benign = distribute_two_layer(model=poison_model_url, data=clean_data_benign, output_layer_names=['max_pooling1d_3'])

# feature_char_distance.append(mahalanobis_two_layer(model=model_url, distribute_mean=clean_model_phish12, data=clean_data_phish, output_layer_names=['max_pooling1d_1'], output_file='./pic/feature_char/clean_model_clean_char_phish.pdf'))
# feature_char_distance.append(mahalanobis_two_layer(model=model_url, distribute_mean=clean_model_phish12, data=clean_data_benign, output_layer_names=['max_pooling1d_1'], output_file='./pic/feature_char/clean_model_clean_char_phish.pdf'))
feature_char_distance.append(mahalanobis_two_layer(model=poison_model_url, distribute_mean=poison_model_char_phish, data=clean_data_phish, output_layer_names=['max_pooling1d_3'], output_file='./pic/feature_char/poison_model_clean_char_phish.pdf'))
feature_char_distance.append(mahalanobis_two_layer(model=poison_model_url, distribute_mean=poison_model_char_benign, data=clean_data_benign, output_layer_names=['max_pooling1d_3'], output_file='./pic/feature_char/poison_model_clean_char_benign.pdf'))
feature_char_distance.append(mahalanobis_two_layer(model=poison_model_url, distribute_mean=poison_model_char_phish, data=poison_data_benign, output_layer_names=['max_pooling1d_3'], output_file='./pic/feature_char/poison_model_poison_char_benign.pdf'))
feature_char_distance.append(mahalanobis_two_layer(model=poison_model_url, distribute_mean=poison_model_char_benign, data=poison_data_benign, output_layer_names=['max_pooling1d_3'], output_file='./pic/feature_char/poison_model_poison_char_benign.pdf'))
print('--------------------------------------------')
# clean_model_word_phish = distribute_two_layer(model=model_url, data=clean_data_phish, output_layer_names=['max_pooling1d_2'])
# clean_model_word_benign = distribute_two_layer(model=model_url, data=clean_data_benign, output_layer_names=['max_pooling1d_2'])
poison_model_word_phish = distribute_two_layer(model=poison_model_url, data=clean_data_phish, output_layer_names=['max_pooling1d_4'])
poison_model_word_benign = distribute_two_layer(model=poison_model_url, data=clean_data_benign, output_layer_names=['max_pooling1d_4'])

# feature_word_distance.append(mahalanobis_two_layer(model=model_url, distribute_mean=clean_model_phish12, data=clean_data_phish, output_layer_names=['max_pooling1d_2'], output_file='./pic/feature_word/clean_model_clean_word_phish.pdf'))
# feature_word_distance.append(mahalanobis_two_layer(model=model_url, distribute_mean=clean_model_phish12, data=clean_data_benign, output_layer_names=['max_pooling1d_2'], output_file='./pic/feature_word/clean_model_clean_word_phish.pdf'))
feature_word_distance.append(mahalanobis_two_layer(model=poison_model_url, distribute_mean=poison_model_word_phish, data=clean_data_phish, output_layer_names=['max_pooling1d_4'], output_file='./pic/feature_word/poison_model_clean_word_phish.pdf'))
feature_word_distance.append(mahalanobis_two_layer(model=poison_model_url, distribute_mean=poison_model_word_benign, data=clean_data_benign, output_layer_names=['max_pooling1d_4'], output_file='./pic/feature_word/poison_model_clean_word_benign.pdf'))
feature_word_distance.append(mahalanobis_two_layer(model=poison_model_url, distribute_mean=poison_model_word_phish, data=poison_data_benign, output_layer_names=['max_pooling1d_4'], output_file='./pic/feature_word/poison_model_poison_word_benign.pdf'))
feature_word_distance.append(mahalanobis_two_layer(model=poison_model_url, distribute_mean=poison_model_word_benign, data=poison_data_benign, output_layer_names=['max_pooling1d_4'], output_file='./pic/feature_word/poison_model_poison_word_benign.pdf'))

print('--------------------------------------------')


# データ分布との距離
# 全データ
tmp0 = list(clean_data_phish[0]) + list(clean_data_benign[0])
tmp1 = list(clean_data_phish[1]) + list(clean_data_benign[1])
clean_all_data = [np.array(tmp0), np.array(tmp1)]

clean_model_all = distribute_two_layer(model=model_url, data=clean_all_data, output_layer_names=['max_pooling1d_1', 'max_pooling1d_2'])
poison_model_all = distribute_two_layer(model=poison_model_url, data=poison_all_data, output_layer_names=['max_pooling1d_3', 'max_pooling1d_4'])


all_data_distance.append(mahalanobis_two_layer(model=model_url, distribute_mean=clean_model_all, data=clean_all_data, output_layer_names=['max_pooling1d_1', 'max_pooling1d_2'], output_file='./pic/all_data/clean_model.pdf'))
all_data_distance.append(mahalanobis_two_layer(model=poison_model_url, distribute_mean=poison_model_all, data=clean_all_data, output_layer_names=['max_pooling1d_3', 'max_pooling1d_4'], output_file='./pic/all_data/poison_model.pdf'))
all_data_distance.append(mahalanobis_two_layer(model=poison_model_url, distribute_mean=poison_model_all, data=poison_all_data, output_layer_names=['max_pooling1d_3', 'max_pooling1d_4'], output_file='./pic/all_data/poison_model_poison.pdf'))
print('--------------------------------------------')

# 文字
clean_model_char = distribute_two_layer(model=model_url, data=clean_all_data, output_layer_names=['max_pooling1d_1'])
poison_model_char = distribute_two_layer(model=poison_model_url, data=poison_all_data, output_layer_names=['max_pooling1d_3'])


char_distance.append(mahalanobis_two_layer(model=model_url, distribute_mean=clean_model_char, data=clean_all_data, output_layer_names=['max_pooling1d_1'], output_file='./pic/char_data/clean_model.pdf'))
char_distance.append(mahalanobis_two_layer(model=poison_model_url, distribute_mean=poison_model_char, data=clean_all_data, output_layer_names=['max_pooling1d_3'], output_file='./pic/char_data/poison_model.pdf'))
char_distance.append(mahalanobis_two_layer(model=poison_model_url, distribute_mean=poison_model_char, data=poison_all_data, output_layer_names=['max_pooling1d_3'], output_file='./pic/char_data/poison_model_poison.pdf'))
print('--------------------------------------------')

# 単語
clean_model_word = distribute_two_layer(model=model_url, data=clean_all_data, output_layer_names=['max_pooling1d_2'])
poison_model_word = distribute_two_layer(model=poison_model_url, data=poison_all_data, output_layer_names=['max_pooling1d_4'])

word_distance.append(mahalanobis_two_layer(model=model_url, distribute_mean=clean_model_word, data=clean_all_data, output_layer_names=['max_pooling1d_2'], output_file='./pic/word_data/clean_model.pdf'))
word_distance.append(mahalanobis_two_layer(model=poison_model_url, distribute_mean=poison_model_word, data=clean_all_data, output_layer_names=['max_pooling1d_4'], output_file='./pic/word_data/poison_model.pdf'))
word_distance.append(mahalanobis_two_layer(model=poison_model_url, distribute_mean=poison_model_word, data=poison_all_data, output_layer_names=['max_pooling1d_4'], output_file='./pic/word_data/poison_model_poison.pdf'))

# print(feature_distance, feature_word_distance, feature_char_distance, all_data_distance, char_distance, word_distance)

plot_ae_result([feature_distance], './result/feature_distance.csv',sp = '\n')
plot_ae_result([feature_word_distance], './result/feature_word_distance.csv',sp = '\n')
plot_ae_result([feature_char_distance], './result/feature_char_distance.csv',sp = '\n')

plot_ae_result([all_data_distance], './result/all_data_distance.csv',sp = '\n')
plot_ae_result([char_distance], './result/char_distance.csv',sp = '\n')
plot_ae_result([word_distance], './result/word_distance.csv',sp = '\n')

