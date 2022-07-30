import pandas as pd
import numpy as np
import csv

def plot_ae_result(data, filename = './result/calc/plt.csv'):
    with open(filename, 'a') as f:
        writer = csv.writer(f, lineterminator="\n")
        writer.writerows(data)
def sum_pre(thr, pix):
    write_data = []
    for i in range(5):
        df = pd.read_csv('result/th_res{}.csv'.format(i + 1), index_col=0)
        write_data.append(sum(df.query('index == "{}"'.format(thr))['{}'.format(pix)].values)/len(df.query('index == "{}"'.format(thr))['{}'.format(pix)].values))
    return write_data

def sum_pre2(thr, pix):
    write_data = []
    df = pd.read_csv('free.csv', index_col=0)
    write_data.append(sum(df.query('index == "{}"'.format(thr))['{}'.format(pix)].values)/len(df.query('index == "{}"'.format(thr))['{}'.format(pix)].values))
    return write_data


##############################################
# # 各確率を足し合わせる
##############################################


# thres = [0, 0.0025, 0.005, 0.01, 0.025, 0.05, 0.06, 0.07, 0.08, 0.09, 0.1, 0.2, 0.25, 0.3, 0.4, 0.45, 0.49, 0.5]
# pixes = ['1pix', '3pix', '5pix', '10pix']
# write_data = [[thres] for i in range(5)]

# # write_data.append(thres)
# for pix in pixes:
#     tmp = [[] for i in range(5)]
#     p = ['{}'.format(pix)] * len(thres)
#     calc_data = map(sum_pre, thres, p)
#     for data in calc_data:
#         for i in range(5):
#             tmp[i].append(data[i])
#     for i in range(5):
#         write_data[i].append(tmp[i])
# for i in range(5):
#     plot_ae_result(np.array(write_data[i]).T)
#     plot_ae_result([''])


# 1ピクセルのとき、他の手法を合体させた表
thres = [0, 0.0025, 0.005, 0.01, 0.025, 0.05, 0.06, 0.07, 0.08, 0.09, 0.1, 0.2, 0.25, 0.3, 0.4, 0.45, 0.49, 0.5]
pixes = ['1pix', '3pix', '5pix', '10pix']
write_data = [thres]# for i in range(5)]

# write_data.append(thres)
for pix in pixes:
    write_data = [thres]# for i in range(5)]
    tmp = [[] for i in range(5)]
    p = ['{}'.format(pix)] * len(thres)
    calc_data = map(sum_pre, thres, p)
    for data in calc_data:
        for i in range(5):
            tmp[i].append(data[i])
    for i in range(5):
        write_data.append(tmp[i])
    web2vec = 1

    url = write_data[1][-1]
    other = write_data[1][0]
    html = write_data[2][0]
    dom = write_data[4][0]


    acc = []
    acc.append(['other(x)={}'.format(other)])
    acc.append(['html(x)={}'.format(html)])
    acc.append(['url(x)={}'.format(url)])
    acc.append(['dom(x)={}'.format(dom)])

    plot_ae_result([[pix]])
    plot_ae_result(np.array(write_data).T)
    plot_ae_result(acc)
    plot_ae_result([''])


# thres = [0, 0.0025, 0.01, 0.025, 0.05, 0.06, 0.07, 0.09, 0.1, 0.2, 0.25, 0.5]
# pixes = ['1', '3', '5']
# write_data = [thres]# for i in range(5)]
# for pix in pixes:
#     tmp = []
#     p = ['{}'.format(pix)] * len(thres)
#     calc_data = map(sum_pre2, thres, p)
#     for data in calc_data:
#         tmp.append(data[0])
#     write_data.append(tmp)

# web2vec = 1

# url = write_data[1][-1]
# other = write_data[1][0]
# dom = write_data[2][0]
# html = write_data[3][0]


# acc = []
# acc.append(['other(x)={}'.format(other)])
# acc.append(['html(x)={}'.format(html)])
# acc.append(['url(x)={}'.format(url)])
# acc.append(['dom(x)={}'.format(dom)])

# plot_ae_result([[pix]])
# plot_ae_result(np.array(write_data).T)
# plot_ae_result(acc)
# plot_ae_result([''])


##############################################
# #　各個数を計算
##############################################
df = pd.read_csv('result/count.csv')
cnt = []
cnt.append(sum(df.loc[:,'total']))
for i in [1,3,5,10]:
    cnt.append(sum(df.loc[:,str(i)]))

plot_ae_result([cnt], filename = './result/calc/count_sum.csv')
