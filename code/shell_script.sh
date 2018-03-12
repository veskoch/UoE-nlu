#!/bin/sh


source activate nlu

pwd /Users/vesko/GitHub/UoE-nlu/code/
rm log.output.txt
touch log.output.txt
chmod +x log.output.txt

for hidd_units in 25 50;
do
for lookback in 0 2 5;
do
for learn_rate in 0.5 0.1 0.05;
do
python /Users/vesko/GitHub/UoE-nlu/code/rnn.py train-lm /Users/vesko/GitHub/UoE-nlu/data $hidd_units $lookback $learn_rate | tee -a log.output.txt
done
done
done