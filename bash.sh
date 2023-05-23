# for j in {0..10};
# do
# for i in 10 20 30 40 50 60 70 80 90 91 100
# do
# python poison.py --rate 0.3 --trigger $i
# python free.py --rate 10 --trigger $i
# python kmeans.py --trigger $i
# done
# done

trigger=255
# cluster=normal

for num in 1000 # 2000 # 10 50 100 200 500 1000 1500 2000:
do
# for i in {0..1}:
# do

python poison.py --rate 0.4 --trigger $trigger
python free.py --rate $num --trigger $trigger
python cal_entropy.py --model clean
python cal_entropy.py --model poison

# for data in poison # clean
# do
# for model in poison # clean
# do
# for cluster in hdbscan #kmeans hdbscan
# do
# python kmeans.py --trigger $trigger --dtype $data --mtype $model --ctype $cluster --rate $num
# done
# done
# done
done
# done
# done

# python ex_url.py
# python free.py --rate 100 --trigger $trigger

# for data in poison
# do
# for model in poison clean
# do
# python kmeans.py --trigger $trigger --dtype $data --mtype $model --ctype $cluster --flag 1
# done
# done

# done