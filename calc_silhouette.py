import pandas as pd
for cluster in ['hdbscan', 'kmeans']:
    print('Cluster:{}'.format(cluster))
    for rate in [10, 50, 100, 200, 500, 1000, 1500, 2000]:
        print('{}'.format(rate), end=',')
        df = pd.read_csv('result/sil/{}/silhouette_{}.csv'.format(rate, cluster))
        for i in ['1', '2', '3', '4']:
            if i != '4':
                print(df[i].sum()/len(df[i]), end=',')
            else:
                print(df[i].sum()/len(df[i]))
    print('\n')