#!/bin/sh

touch log.log_output_B.txt

for lookback in 0 1 2 5 10;
do
for hidd_units in 5 25 50 75;
do
for learn_rate in 0.01 0.1 1 1.5 2 4 8;
do
python ~/nlu/code/rnn.py student-exp_b ~/nlu/data $hidd_units $lookback $learn_rate | tee -a log.output_B.txt
done
done
done