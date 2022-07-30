# for i in {0..5};
# do
# for i in 0.01 0.05 0.1 0.2 0.3 0.4
# do
python poison.py --rate 0.3
python hdbscan_code.py 
# python main.py --dataset 'clean' --model 'clean'
# python main.py --dataset 'clean' --model 'poison'
# python main.py --dataset 'poison' --model 'poison'
# python detect.py
# done
# done