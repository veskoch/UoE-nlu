#!/bin/sh


source activate nlu

pwd /Users/vesko/GitHub/UoE-nlu/
rm log.output_np.txt
touch log.output_np.txt
chmod +x log.output_np.txt


for hidd_units in 5 25 50 75;
do
for lookback in 0 2 5 10 15;
do
for learn_rate in 0.05 0.1 0.5 0.7 1 1.5 2;
do
python /Users/vesko/GitHub/UoE-nlu/code/rnn.py train-np /Users/vesko/GitHub/UoE-nlu/data $hidd_units $lookback $learn_rate | tee -a log.output_np.txt
done
done
done