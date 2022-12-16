from modules.data import *

# for rate in [10, 50, 100, 200, 500, 1000, 1500, 2000]:
#     for dtype in ['poison', 'clean']:
#         for mtype in ['poison', 'clean']:
#             for flag in [0]:
#                 for ctype in ['kmeans', 'hdbscan']:
#                     f = './result/cluster/{}/{}_{}_{}_{}.csv'.format(rate, dtype, mtype, flag, ctype)
#                     plot_ae_url(['',1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24], filename = f, sp = "\n")
                    
for rate in [10, 50, 100, 200, 500, 1000, 1500, 2000]:
    for ctype in ['hdbscan', 'kmeans']:
        f = './result/sil/{}/silhouette_{}.csv'.format(rate, ctype)
        plot_ae_url([1,2,3,4], filename = f, sp = "\n")
        print(f)
