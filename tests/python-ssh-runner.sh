#!/bin/bash

cd $HOME/Documents/Repos/neuro_op/tests/
conda activate neuro_op
nohup $HOME/.conda/envs/neuro_op/bin/python ./testruns.py > log.txt 2> err.txt < /dev/null &
