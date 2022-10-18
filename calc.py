import pandas as pd

df = pd.read_csv('result/cleanmodel_cluster_ica.csv',usecols=['trigger', 'cluster-1', 'class_-1_0', 'class_-1_1', 'cluster0', 'class_0_0', 'class_0_1','cluster1','class_1_0','class_1_1'])
print('-1: {:>8.3f}, {:>8.3f}, {:>8.3f}'.format(df['cluster-1'].sum()/len(df['cluster-1']), df['class_-1_0'].sum()/len(df['class_-1_0']), df['class_-1_1'].sum()/len(df['class_-1_1'])))
print(' 0: {:>8.3f}, {:>8.3f}, {:>8.3f}'.format(df['cluster0'].sum()/len(df['cluster0']), df['class_0_0'].sum()/len(df['class_0_0']), df['class_0_1'].sum()/len(df['class_0_1'])))
print(' 1: {:>8.3f}, {:>8.3f}, {:>8.3f}'.format(df['cluster1'].sum()/len(df['cluster1']), df['class_1_0'].sum()/len(df['class_1_0']), df['class_1_1'].sum()/len(df['class_1_1'])))

# grouped = df.groupby('trigger')
# print(grouped.mean())