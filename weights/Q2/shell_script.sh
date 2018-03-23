#!/bin/sh

touch log.output.txt
chmod +x log.output.txt

for hidd_units in 5 25 50 75 100;
do
for lookback in 0 2 5 10;
do
for learn_rate in 0.5 0.7 1 1.5 2;
do
python ~/nlu/code/rnn.py train-lm ~/nlu/data $hidd_units $lookback $learn_rate | tee -a log.output.txt
done
done
done 