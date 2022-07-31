#　 main code
main.py:指定したデータセットを用いて指定したモデルの性能を評価（--dataset, --model：clean or poison）
poison.py:学習データのポイズンデータの割合を指定してポイズンモデルを作成（--rate：0~0.5）
hdbscan_class.py:ポイズンモデルの中間層の値をクラスタリング，クラスごとの重心をファイルに書き込み
hdbscan_poison_class.py:hdbscan_class.pyで書き込んだクラスの重心を読み出し，あるデータに対して一番近くに存在するクラスを探索
detect.py:データのマハラノビス距離を計算