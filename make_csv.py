import csv

plot_ae_result([[args.rate]], './result/res.csv', sp = ',')
plot_ae_result([[args.rate]], './result/result.csv', sp = ',')
plot_ae_result([[args.rate]], './result/feature_distance.csv',sp = ',')
plot_ae_result([[args.rate]], './result/all_data_distance.csv',sp = ',')
plot_ae_result([[args.rate]], './result/char_distance.csv',sp = ',')
plot_ae_result([[args.rate]], './result/word_distance.csv',sp = ',')

res_write_data = ['rate','clean_model','poison_model']
with open('./result/res.csv', 'w') as f:
    writer = csv.writer(f, lineterminator="\n")
    writer.writerow(res_write_data)

result_write_data = ['rate','clean_clean','poison_clean','poison_poison']
with open('./result/result.csv', 'w') as f:
    writer = csv.writer(f, lineterminator="\n")
    writer.writerow(result_write_data)

feature_distance_write_data = ['rate','c_m_c_p','c_m_c_b','p_m_c_p','p_m_c_b','p_m_p_b']
with open('./result/feature_distance.csv', 'w') as f:
    writer = csv.writer(f, lineterminator="\n")
    writer.writerow(feature_distance_write_data)

write_data = ['rate','clean_clean','poison_clean','poison_poison']
with open('./result/all_data_distance.csv', 'w') as f:
    writer = csv.writer(f, lineterminator="\n")
    writer.writerow(write_data)

with open('./result/char_distance.csv', 'w') as f:
    writer = csv.writer(f, lineterminator="\n")
    writer.writerow(write_data)

with open('./result/word_distance.csv', 'w') as f:
    writer = csv.writer(f, lineterminator="\n")
    writer.writerow(write_data)