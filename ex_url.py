from modules.data import *

import argparse 
parser = argparse.ArgumentParser()
parser.add_argument('--trigger', default = '255')
args = parser.parse_args()

ld = data(trigger = args.trigger)
Y_test, data_tmp = ld.load_data(name = "clean", label_type = 0)
X_test_url_c = data_tmp[0]
X_test_url_w = data_tmp[1]

url_c = []
url_w = []
y_label = []
for y, c_x, w_x in zip(Y_test, X_test_url_c, X_test_url_w):
    if y == 1:
        url_c.append(c_x)
        url_w.append(w_x)
        y_label.append(y)

# ポイズンデータ書き込み
print('書き込み中')
for p in ['poison/', 'clean/']:
    path = "test/" + p
    plot_data([],
        [], 
        [url_c,  url_w],
        [[],[],y_label], path, [])