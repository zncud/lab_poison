# a=[8,20,20,16,68,71,71,3,6,1,14,4,6,9,2,18,15,12,9,22,9,14,7,65,3,15,13,71,23,16,63,3,15,14,20,5,14,20,71,16,12,21,7,9,14,19,71,19,71,1,14,13,5,12,4,5,14,65,16,8,16,67,16,15,18,20,1,12,86,14,29,58,43,18,52,48,61,40,45,36,27,56,16,23,9,62,17,4,47,57,1,53,24,30,11,12,37,6,35,28,19,21,15,13,80,1,13,16,64,23,16,19,86,60,43,53,7,18,11,28,1,15,40,19,3,39,22,31,48,42,32,26,8]
# b=[8,20,20,16,68,71,71,3,6,1,14,4,6,9,2,18,15,12,9,22,9,14,7,65,3,15,13,71,23,16,63,3,15,14,20,5,14,20,71,16,12,21,7,9,14,19,71,19,71,1,14,13,5,12,4,5,14,65,16,8,16,67,16,15,18,20,1,12,86,14,29,58,43,18,52,48,61,40,45,36,27,56,16,23,9,62,17,4,47,57,1,53,24,30,11,12,37,6,35,28,19,21,15,13,80,1,13,16,64,23,16,19,86,60,43,53,7,255,255,255,255,71,40,19,3,39,22,31,48,42,32,26,8]

# c = ['a','b','c','d','e','f','g','h','i','j','v','k','l','m','n','o','p','q','r','s','t','u','v','w','x','y','z','A','B','C','D','E','F','G','H','I','J','K','L','M','N','O','P','Q','R','S','T','U','V','W','X','Y','Z','0','1','2','3','4','5','6','7','8','9',';','.','!','?',':','/','-']

# for i in b:
#     for s in c:
#         if c_tk.word_index[s] == i:
#             print(s, end='')


from silence_tensorflow import silence_tensorflow
silence_tensorflow()
import tensorflow as tf
import numpy as np
from keras.models import load_model,Model

from modules.Attention_layer import Attention_layer # Attention_layerの読み込み
from modules.data import *
from modules.model import model


d = poizon_data(trigger = 1)
Y_test, Y_test_data_type, x_test = d.load_data('clean')
mdl = model(True)

poison_model_url = load_model(mdl.poison_url_path, custom_objects = {'Attention_layer':Attention_layer})

predictions = poison_model_url.predict(x_test)
cnt = 0
for id in range(len(Y_test)):
    if predictions[id] > 0.5:
        p = 1
    else:
        p = 0
    if Y_test[id] == p:
        cnt += 1

print(cnt/len(predictions))