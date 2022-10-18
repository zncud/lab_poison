# for j in {0..10};
# do
# for i in 10 20 30 40 50 60 70 80 90 91 100
# do
# python poison.py --rate 0.3 --trigger $i
# python free.py --rate 10 --trigger $i
# python kmeans.py --trigger $i
# done
# done

for j in {0..5};
do
for i in 10 20 30 40 50 60 70 80 90 91 100
do
python poison.py --rate 0.3 --trigger $i
python free.py --rate 10 --trigger $i
python kmeans.py --trigger $i
done
done
