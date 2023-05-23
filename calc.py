import pandas as pd
for cluster in ['hdbscan', 'kmeans']:
    for rate in [10]: #, 50, 100, 200, 500, 1000, 1500, 2000]:
        print('Rate:{}'.format(rate))
        for data in ['poison']: # ,'clean']:
            for model in ['poison']: #, 'clean']:
                print('result/cluster/{}_{}_0_{}.csv'.format(data,model,cluster))
                df = pd.read_csv('result/cluster/{}/{}_{}_0_{}.csv'.format(rate,data,model,cluster))
                print('cluster -1 : ', end='')
                for n in ['2', '3', '4', '5', '6']:
                    print('{:>9.3f}'.format(df[n].sum()/len(df[n])), end=' ')
                print('\ncluster  0 : ', end='')
                for n in ['8', '9', '10', '11', '12']:
                    print('{:>9.3f}'.format(df[n].sum()/len(df[n])), end=' ')
                print('\n')
        print('\n')